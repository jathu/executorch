// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250317"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "91ed5dba72d208e857a4619e835341250a79ca795b698eb0c72516513711e8b1",
    "sha256" + debug: "80d3226169e38d9b69639c686779b8e1b7f2672f5e5cc6b03e327509053bf632",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b4b0d01204efc30a74103c60e1731120b48d60c5931e1b5a5facd1ea8152ba45",
    "sha256" + debug: "a85502e67193093261a1353fd96cb9beaddc79aea73a6c329c7ea9c3fefe67bc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "927c5333eaac069c136f5a7433296059803936932ead89831079424559ea47b3",
    "sha256" + debug: "c6b9764eda0543774022a0b6208d499c28c19ad9efabe94a3d2922415613e24a",
  ],
  "executorch": [
    "sha256": "aa83fa998b917de0e92bd5ad81df0c9f1e66c159cd3a69b12d552e7106c6dec0",
    "sha256" + debug: "72b040d91be767fc7fa92b502bc9549db9aea92bd770b4e48cf28c044f873026",
  ],
  "kernels_custom": [
    "sha256": "d2c650fda120ccb678e24de04781ed35d4bb73e77ca52c0cdb91280e517e2609",
    "sha256" + debug: "b2b474db25fe39165de004a0effa659d9bb54349b36ba09f40626165a90ae73b",
  ],
  "kernels_optimized": [
    "sha256": "2b9e0e2fc2aebb6bea9c25909805f7ed70bd1932fd0e3882a74ca265869220c4",
    "sha256" + debug: "95a90eafcac682dbfe368b66d825712a4c68eff8638afc29e0af7334714604ad",
  ],
  "kernels_portable": [
    "sha256": "90a01d56da5cd1ca8d39d5404735c162e877d1e04d4fd470fc93d4a39dba802f",
    "sha256" + debug: "fe2470aef9bd316c2600ada3612ceeeed068d83d12a38439b7c4414f82443747",
  ],
  "kernels_quantized": [
    "sha256": "a60ff1239761f95fd4413f13c4c672b91d16adb6aeb3b0ac55d145af659cb989",
    "sha256" + debug: "60c7cb345bde3091481a6b92ef111ade11454fc5bbfd4792b2595aad8bced498",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
