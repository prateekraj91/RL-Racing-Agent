# Asset Licenses and Attribution

All external 3D, texture, and environmental assets included in this repository are zero-cost, open-access, and clean-licensed under Creative Commons CC0 1.0 Universal (Public Domain Dedication).

---

## 1. Environment & Skybox (HDRI)

* **Asset**: `kloofendal_48d_partly_cloudy_puresky_2k.hdr`
* **Source**: [Poly Haven](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
* **Authors**: Greg Zaal & Andreas Mischok
* **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
* **Format**: 2048 x 1024 high dynamic range image (.hdr)
* **Usage**: Environment map, scene lighting reflections, and background dome.

---

## 2. Road Surface PBR Textures

* **Asset**: `asphalt_02` (`asphalt_diff_1k.jpg`, `asphalt_rough_1k.jpg`, `asphalt_nor_1k.jpg`, `asphalt_diff_2k.jpg`, `asphalt_rough_2k.jpg`, `asphalt_nor_2k.jpg`)
* **Source**: [Poly Haven](https://polyhaven.com/a/asphalt_02)
* **Author**: Rob Tuytel
* **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
* **Format**: 1K and 2K JPEG textures (Albedo/Diffuse, Roughness, OpenGL Normal)
* **Usage**: PBR surface material for track ribbon mesh.

---

## 3. Ground / Runoff PBR Textures

* **Asset**: `aerial_grass_rock` (`ground_diff_1k.jpg`, `ground_rough_1k.jpg`, `ground_nor_1k.jpg`)
* **Source**: [Poly Haven](https://polyhaven.com/a/aerial_grass_rock)
* **Author**: Rob Tuytel
* **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
* **Format**: 1K JPEG textures (Albedo/Diffuse, Roughness, OpenGL Normal)
* **Usage**: Ground runoff plane surrounding the circuit.

---

## 4. Vehicle 3D Model

* **Asset**: Procedural (no external model file).
* **Source**: Parametric Three.js geometry ported from `docs/concept_reference/main.js`
  (the project's own concept render) — a generic, unbranded stylized sports coupe.
* **Author**: RL Racing Agent Team.
* **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
* **Note**: A previous `car.glb` (a third-party, branded sports-car model under CC-BY)
  was removed; no branded vehicle model, texture, logo, or badge ships in this repo.
* **Usage**: Autonomous racing agent vehicle, built in `frontend/app.js`
  (`buildProceduralVehicle`), wheelbase scaled to the env's 50 px.

---

## 5. Vendor Libraries (JavaScript)

* **Three.js**: Version r128 (MIT License - Copyright 2010-2021 Three.js Authors)
* **OrbitControls**: MIT License
* **RGBELoader**: MIT License
* **GLTFLoader**: MIT License
* **DRACOLoader**: MIT License
* **Draco Decoder**: Google Draco 3D data compression library (Apache License 2.0)
* **EffectComposer & Passes**: RenderPass, ShaderPass, UnrealBloomPass, SSAOPass, SMAAPass, CopyShader, LuminosityHighPassShader, SSAOShader, SMAAShader, SimplexNoise (MIT License - Three.js Examples)

---

## 6. Procedural Environment Props Architecture

* **Source**: Parametric 3D meshes generated in Three.js (inspired by Kenney & Quaternius CC0 modular motorsport/nature assets)
* **Author**: RL Racing Agent Team
* **License**: [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
* **Technique**: High-efficiency GPU instancing (`THREE.InstancedMesh`)
* **Components**:
  * **Trees**: Instanced pine (cone) + round (icosahedron) canopies with per-instance
    green tint and a shared trunk, scattered outside the track.
  * **Skyline**: Instanced windowed blue-grey towers on the far horizon.
  * **Grandstand + Pit**: Tiered crowd stand, roof, pillars and a pit building with a
    cyan light band, along the start straight.
* **Spatial Constraint**: Trees are kept $\ge 150\text{ px}$ from the centerline and every
  scenery prop $\ge 100\text{ px}$, strictly preserving the $70\text{ px}$ drivable
  corridor (asserted by `tests/e2e/test_demo_smoke.py::test_scenery_props_clear_of_centerline`).
