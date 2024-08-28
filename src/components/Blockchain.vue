<script>
import Stats from "stats.js";
import { defineAsyncComponent } from "vue";
import axios from "axios";
import * as THREE from "three";
// non fanno più parte della libreria bisogna importarli manualemnte
import { FontLoader } from "three/examples/jsm/loaders/FontLoader.js";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";
export default {
  name: "Blockchain",
  components: {
    block: defineAsyncComponent(() => import("./Block.vue")), // lazy loader viene caricata solo quando si ha bisogno del componente
  },
  data() {
    return {
      blocks: [],
      show_block: false,
      block_iesimo: "",
    };
  },
  methods: {
    async call() {
      try {
        // const data = await axios("https://blockstream.info/api/blocks/tip/height"); // altezza ultimo blocco
        const data = await axios("https://blockstream.info/api/blocks/");
        this.blocks = data.data;
      } catch (error) {
        console.log(error.data);
      }
    },
    async GetBlock(i) {
      this.show_block = true;

      const blockHash = await axios(`https://blockstream.info/api/block-height/${i}`);
      const dataBlock = await axios(`https://api.blockcypher.com/v1/btc/main/blocks/${blockHash.data}`);

      this.block_iesimo = dataBlock.data;
      console.log(this.block_iesimo);

      // difficolta è quanto è difficile minare un blocco, un aumento del target diminuisce la difficolta, una diminuzione del target aumenta la difficolta
      // bits è la forma compatta del target
      // size dimensione del blocco in bytes
      // transection count quantita di transazioni
    },
    blockchain() {
      this.show_block = false;
    },
    new_block() {},
  },
  async mounted() {
    // const stats = new Stats();
    // stats.showPanel(1);
    this.new_block();
    await this.call();
    const space = document.getElementById("container_blockchain");
    // document.body.appendChild(stats.dom);
    // const tick = () => {
    //   stats.begin();
    const scene = new THREE.Scene(); // * contenitore per oggetti 3d
    const camera = new THREE.PerspectiveCamera(75, space.clientWidth / space.clientHeight, 0.1, 1000); // * punto di vista
    // fov estensione della scena, aspect ratio, near, far  is that objects further away from the camera than the value of far or closer than near won't be rendered.

    const directionalLight = new THREE.DirectionalLight(0xffffff, 3);
    directionalLight.position.set(1, 1, 10); // Posizionamento della luce asse x y z più è alto lo z e più sembrera che venga dalla telecamera
    scene.add(directionalLight);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true }); // * generatore immagini
    renderer.setSize(space.clientWidth, space.clientHeight); // spazio immagini
    space.appendChild(renderer.domElement);

    let distance = 18;
    const numbblock = 10;
    let flag = true;
    let half_cubes = Math.floor(numbblock / 2);

    const geometry = new THREE.BoxGeometry(1.4, 1.4, 1.4); // dimensioni block
    let material = new THREE.MeshStandardMaterial({
      // materiale blocco
      color: new THREE.Color(0x8a8a8a),
      metalness: 1, //  metallicità
      roughness: 0.3, //  ruvidità
      emissive: new THREE.Color(0, 0, 0),
    });
    const cube = new THREE.Mesh(geometry, material); // prende una geometria e l'applica al materiale

    const numb_mat = new THREE.MeshStandardMaterial({
      // materiale testo
      color: new THREE.Color(0xff0000),
      metalness: 1, //  metallicità
      roughness: 0.3, //  ruvidità
      emissive: new THREE.Color(0, 0, 0),
    });
    for (let i = 0; i < this.blocks.length; i++) {
      if (i >= half_cubes) flag = false;
      if (flag) cube.position.set(-2, distance - numbblock, -1);
      else cube.position.set(2, distance, -1);
      distance -= 2;
      scene.add(cube.clone());
      // text
      const loader = new FontLoader();
      loader.load("../../node_modules/three/examples/fonts/gentilis_bold.typeface.json", (font) => {
        // impossibile centrare due linee a meno che non si crea una mesh per ogni linea. You could create a geometry for each line, perform the centering and then merge the geometries into a single one. Would this tradeoff be acceptable to you?
        // TESTO

        const numb_geometry = new TextGeometry(this.blocks[i].height.toString(), {
          font: font,
          size: 0.15,
          depth: 0.7, // inizia dal centro del cubo
        });
        const string_geometry = new TextGeometry("Number Block", {
          font: font,
          size: 0.1,
          depth: 0.7, // inizia dal centro del cubo
        });
        const meshs = [];
        for (let j = 0; j < 4; j++) {
          const numb_mesh = new THREE.Mesh(numb_geometry, numb_mat);
          const string_mesh = new THREE.Mesh(string_geometry, numb_mat);
          meshs.push({ NUMB_MESH: numb_mesh, STRING_MESH: string_mesh });
          scene.children[i + 1].add(numb_mesh, string_mesh);
        }

        const size_number = size(meshs[0].NUMB_MESH);
        const size_string = size(meshs[0].STRING_MESH);

        // cordinate testo su cubo
        meshs[0].NUMB_MESH.position.set(-(size_number.x / 2), -size_number.y / 2, 0.05); // Faccia frontale
        meshs[0].STRING_MESH.position.set(-0.4, -size_number.y / 2 + 0.3, 0.05);

        meshs[1].NUMB_MESH.position.set(size_number.x / 2, -size_number.y / 2, -0.05); // Faccia posteriore
        meshs[1].STRING_MESH.position.set(0.4, -size_number.y / 2 + 0.3, -0.05);
        meshs[1].NUMB_MESH.rotation.y = meshs[1].STRING_MESH.rotation.y = Math.PI;

        meshs[2].NUMB_MESH.position.set(0.05, -size_number.y / 2, size_number.x / 2); // Faccia destra
        meshs[2].STRING_MESH.position.set(0.05, -size_number.y / 2 + 0.3, size_string.x / 2);
        meshs[2].NUMB_MESH.rotation.y = meshs[2].STRING_MESH.rotation.y = Math.PI / 2;

        meshs[3].NUMB_MESH.position.set(-0.05, -size_number.y / 2, -size_number.x / 2); // Faccia sinistra
        meshs[3].STRING_MESH.position.set(-0.05, -size_number.y / 2 + 0.3, -size_string.x / 2);
        meshs[3].NUMB_MESH.rotation.y = meshs[3].STRING_MESH.rotation.y = -Math.PI / 2;
      });
    }

    function size(ob) {
      const size_mesh = new THREE.Box3().setFromObject(ob);
      const mesh_size = new THREE.Vector3();
      const size_block = size_mesh.getSize(mesh_size);
      return size_block;
    }

    camera.position.z = 6;
    camera.position.y = 4;

    const animation_speed = 0.01;

    const animation = () => {
      // è il render delle immagini.  This will create a loop that causes the renderer to draw the scene every time the screen is refreshed (on a typical screen this means 60 times per second).
      requestAnimationFrame(animation);
      scene.children.forEach((block, y) => {
        block.rotation.y += animation_speed; // rotation or position
      });

      renderer.render(scene, camera);
      renderer.render(scene, camera);
    };
    animation();

    window.addEventListener("resize", () => {
      // responsive
      const width = space.clientWidth;
      const height = space.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
    });
    // stats.end();
    // requestAnimationFrame(tick);
    // };
    // requestAnimationFrame(tick);
    // tick();
  },
};
</script>

<template>
  <main>
    <block v-show="show_block" :value="block_iesimo" @close="blockchain" />
    <div class="container-fluid" v-show="!show_block">
      <div class="row">
        <div id="blocks" class="col-12 col-md-3 text-center">
          <div v-for="(block, i) in blocks">
            <div @click="GetBlock(block.height)" class="py-2 blocks">Altezza Blocco: {{ block.height }}</div>
          </div>
        </div>
        <div id="container_blockchain" class="col-12 col-md-9"></div>
      </div>
    </div>
  </main>
</template>

<style lang="scss">
@use "./../style/general.scss" as *;
#container_blockchain {
  height: calc(100vh - 90px); // 90px header
  position: relative;
  overflow: hidden;
}

.blocks {
  cursor: pointer;
}
</style>
