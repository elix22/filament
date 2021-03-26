/*
 * Copyright (C) 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MorphHelper.h"

#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>

#include "GltfEnums.h"
#include "TangentsJob.h"

using namespace filament;
using namespace filament::math;
using namespace utils;

static constexpr uint8_t kUnused = 0xff;

namespace gltfio {

uint32_t computeBindingSize(const cgltf_accessor* accessor);
uint32_t computeBindingOffset(const cgltf_accessor* accessor);

// Returns true if a is a subset of b.
static bool isSubsetOf(ubyte4 a, ubyte4 b) {
    if (a.x != kUnused && a.x != b.x && a.x != b.y && a.x != b.z && a.x != b.w) return false;
    if (a.y != kUnused && a.y != b.x && a.y != b.y && a.y != b.z && a.y != b.w) return false;
    if (a.z != kUnused && a.z != b.x && a.z != b.y && a.z != b.z && a.z != b.w) return false;
    if (a.w != kUnused && a.w != b.x && a.w != b.y && a.w != b.z && a.w != b.w) return false;
    return true;
}

static bool indexOf(int a, ubyte4 b) {
    if (a == b.x) return 0;
    if (a == b.y) return 1;
    if (a == b.z) return 2;
    if (a == b.w) return 3;
    return -1;
}

MorphHelper::MorphHelper(FFilamentAsset* asset, FFilamentInstance* inst) : mAsset(asset),
        mInstance(inst) {
    NodeMap& sourceNodes = asset->isInstanced() ? asset->mInstances[0]->nodeMap : asset->mNodeMap;
    for (auto pair : sourceNodes) {
        cgltf_node const* node = pair.first;
        cgltf_mesh const* mesh = node->mesh;
        if (mesh) {
            cgltf_primitive const* prims = mesh->primitives;
            for (cgltf_size pi = 0, count = mesh->primitives_count; pi < count; ++pi) {
                addPrimitive(mesh, pi, &mMorphTable[pair.second]);
            }
        }
    }
}

MorphHelper::~MorphHelper() {
    auto engine = mAsset->mEngine;
    for (auto& entry : mMorphTable) {
        for (auto& prim : entry.second.primitives) {
            for (auto& target : prim.targets) {
                engine->destroy(target.bufferObject);
            }
        }
    }
}

void MorphHelper::applyWeights(Entity entity, float const* weights, size_t count) noexcept {
    auto& engine = *mAsset->mEngine;
    auto renderableManager = &engine.getRenderableManager();
    auto renderable = renderableManager->getInstance(entity);

    // If there are 4 or fewer targets, we can simply re-use the original VertexBuffer.
    if (count <= 4) {
        float4 vec{};
        for (size_t i = 0; i < count; i++) {
            vec[i] = weights[i];
        }
        renderableManager->setMorphWeights(renderable, vec);
        return;
    }

    // We honor up to 255 weights because our set representation is a 4-tuple of bytes, with one
    // slot reserved for a sentinel value. Note that 255 is much more than the glTF min spec of 4.
    count = std::min(count, size_t(255));

    // Make a copy of the weights because we want to re-order them.
    auto& sorted = mPartiallySortedWeights;
    sorted.clear();
    sorted.insert(sorted.begin(), weights, weights + count);

    // Find the four highest weights in O(n) by doing a partial sort.
    std::nth_element(sorted.begin(), sorted.begin() + 4, sorted.end(), [](float a, float b) {
        return a > b;
    });

    // Find the "primary indices" which are the indices of the four highest weights. This is O(n).
    ubyte4 primaryIndices = {kUnused, kUnused, kUnused, kUnused};
    for (size_t index = 0, primary = 0; index < count && primary < 4; ++index) {
        const float w = weights[index];
        if (w > 0 && (w == sorted[0] || w == sorted[1] || w == sorted[2] || w == sorted[3])) {
            primaryIndices[primary++] = index;
        }
    }

    // Swap out the buffer objects for the primary indices.
    for (const auto& prim : mMorphTable[entity].primitives) {
        for (const auto& target : prim.targets) {
            int index = indexOf(target.morphTargetIndex, primaryIndices);
            if (index > -1) {
                VertexBuffer* vb = prim.vertexBuffer;
                vb->setBufferObjectAt(engine, target.baseMorphSlot + index, target.bufferObject);
            }
        }
    }

    // Finally, set the 4-tuple uniform for the weight values by derefing the primary indices.
    // Note that we first create a "safe set" by replacing the unused sentinel with zero.
    ubyte4 safe = primaryIndices;
    for (int i = 0; i < 4; i++) {
        if (safe[i] == kUnused) {
            safe[i] = 0;
        }
    }
    float4 highest(weights[safe[0]], weights[safe[1]], weights[safe[2]], weights[safe[3]]);
    renderableManager->setMorphWeights(renderable, highest);
}

void MorphHelper::addPrimitive(cgltf_mesh const* mesh, int primitiveIndex, TableEntry* entry) {
    VertexBuffer* vb = mAsset->mMeshCache.at(mesh)[primitiveIndex].vertices;
    entry->primitives.push_back({ vb });
    std::vector<GltfTarget>& targets = entry->primitives.back().targets;

    const cgltf_primitive& prim = mesh->primitives[primitiveIndex];

    // TODO: This code is a bit fragile; it mimics some of the logic in AssetLoader in order to
    // determine the correct slot indices.
    int slot = 0;
    bool hasNormals = false;
    for (cgltf_size aindex = 0; aindex < prim.attributes_count; aindex++) {
        const cgltf_attribute& attribute = prim.attributes[aindex];
        const int index = attribute.index;
        const cgltf_attribute_type atype = attribute.type;
        const cgltf_accessor* accessor = attribute.data;
        if (atype == cgltf_attribute_type_tangent) {
            continue;
        }
        if (atype == cgltf_attribute_type_normal) {
            slot++;
            hasNormals = true;
            continue;
        }
        slot++;
    }

    // If the model is lit but does not have normals, then we generated flat normals.
    if (prim.material && !prim.material->unlit && !hasNormals) {
        slot++;
    }

    constexpr int baseTangentsAttr = (int) VertexAttribute::MORPH_TANGENTS_0;
    constexpr int basePositionAttr = (int) VertexAttribute::MORPH_POSITION_0;

    for (int targetIndex = 0; targetIndex < prim.targets_count; targetIndex++) {
        const cgltf_morph_target& morphTarget = prim.targets[targetIndex];
        for (cgltf_size aindex = 0; aindex < morphTarget.attributes_count; aindex++) {
            const cgltf_attribute& attribute = morphTarget.attributes[aindex];
            const cgltf_accessor* accessor = attribute.data;
            const cgltf_attribute_type atype = attribute.type;
            if (atype == cgltf_attribute_type_tangent) {
                continue;
            }
            if (atype == cgltf_attribute_type_normal) {
                // TODO: do something here
                slot++;
                continue;
            }
            if (atype == cgltf_attribute_type_position) {
                // TODO: do something here
                slot++;
            }
        }
        // TODO: invoke TangentsJob here
    }

}

}  // namespace gltfio
