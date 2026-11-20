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

* **Asset**: `car.glb`
* **Source**: [Three.js Examples (webgl_materials_car)](https://threejs.org/examples/webgl_materials_car.html) / [GitHub mrdoob/three.js](https://raw.githubusercontent.com/mrdoob/three.js/r128/examples/models/gltf/ferrari.glb)
* **Author**: `vicent091036` (original Ferrari 458 Italia 3D model)
* **License**: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
* **Format**: Binary glTF (.glb) with Draco mesh compression (~1.6 MB)
* **Hierarchy**: Contains named sub-objects `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr`, `steering_wheel`, `body`, `lights_red`.
* **Usage**: Autonomous racing agent vehicle representation in 3D simulator.

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
  * **Pine Trees**: Dual-cone foliage + cylindrical wooden trunk.
  * **Deciduous Oak Trees**: Low-poly dodecahedral canopy + cylindrical trunk.
  * **Stadium / Track Lamp Posts**: Metal poles + emissive warm white fixtures.
  * **Tecpro Runoff Barriers**: High-visibility impact foam cushions (alternating red and white).
  * **Grandstands**: Modular covered spectator grandstands positioned along straightaways.
* **Spatial Constraint**: All environment props are placed at an offset of $\ge 25\text{ px}$ outside track collision barriers, strictly preserving the $70\text{ px}$ drivable corridor.
