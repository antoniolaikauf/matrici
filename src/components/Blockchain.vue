<script>
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
    };
  },
  methods: {
    async call() {
      try {
        // const data = await axios("https://blockstream.info/api/blocks/tip/height"); // altezza ultimo blocco
        const data = await axios("https://blockstream.info/api/blocks/");
        this.blocks = data.data.reverse();
        console.log(this.blocks);
      } catch (error) {
        console.log(error.data);
      }
    },
    async GetBlock(i) {
      this.show_block = true;
      const blockHash = await axios(`https://blockstream.info/api/block-height/${i}`);
      // this.blocks = blockHash.data;
      const dataBlock = await axios(`https://blockstream.info/api/block/${blockHash.data}`); // dati blocco

      // difficolta è quanto è difficile minare un blocco, un aumento del target diminuisce la difficolta, una diminuzione del target aumenta la difficolta
      // bits è la forma compatta del target
      // size dimensione del blocco in bytes
      // transection count quantita di transazioni
    },
    blockchain() {
      this.show_block = false;
    },
  },
  async mounted() {
    await this.call();
    const space = document.getElementById("container_blockchain");

    const scene = new THREE.Scene(); // * contenitore per oggetti 3d
    const camera = new THREE.PerspectiveCamera(75, space.clientWidth / space.clientHeight, 0.1, 1000); // * punto di vista
    // fov estensione della scena, aspect ratio, near, far  is that objects further away from the camera than the value of far or closer than near won't be rendered.

    const directionalLight = new THREE.DirectionalLight(0xffffff, 4);
    directionalLight.position.set(1, 1, 10); // Posizionamento della luce asse x y z più è alto lo z e più sembrera che venga dalla telecamera
    scene.add(directionalLight);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true }); // * generatore immagini
    renderer.setSize(space.clientWidth, space.clientHeight); // spazio immagini
    space.appendChild(renderer.domElement);

    let distance = 0;
    const numbblock = 10;
    let flag = true;
    let half_cubes = Math.floor(numbblock / 2);
    for (let i = 0; i < this.blocks.length; i++) {
      const geometry = new THREE.BoxGeometry(1.3, 1.3, 1.3); // dimensioni block
      let material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x8a8a8a),
        metalness: 1, //  metallicità
        roughness: 0.3, //  ruvidità
        emissive: new THREE.Color(0, 0, 0),
      });

      const cube = new THREE.Mesh(geometry, material); // prende una geometria e l'applica al materiale
      if (i > half_cubes) flag = false;
      if (flag) cube.position.set(2, distance, 0);
      else cube.position.set(-2, distance - half_cubes * 2, 0); // distance - half_cubes * 2 è per diminuire la distanza, se no andrebbe a 12 14 ma riparte da 2 4 ...

      distance += 2;
      // text
      const loader = new FontLoader();
      loader.load("../../node_modules/three/examples/fonts/gentilis_bold.typeface.json", (font) => {
        // impossibile centrare due linee a meno che non si crea una mesh per ogni linea. You could create a geometry for each line, perform the centering and then merge the geometries into a single one. Would this tradeoff be acceptable to you?
        // TESTO

        const numb_geometry = new TextGeometry(this.blocks[i].height.toString(), {
          font: font,
          size: 0.15,
          depth: 0.7, // inizia dal centro del cubo
          align: "center",
        });
        const string_geometry = new TextGeometry("Number Block", {
          font: font,
          size: 0.1,
          depth: 0.7, // inizia dal centro del cubo
        });

        const numb_mat = new THREE.MeshStandardMaterial({
          // materiale testo
          color: new THREE.Color(0xff0000),
          metalness: 1, //  metallicità
          roughness: 0.3, //  ruvidità
          emissive: new THREE.Color(0, 0, 0),
        });

        const meshs = [];
        for (let i = 0; i < 4; i++) {
          const numb_mesh = new THREE.Mesh(numb_geometry, numb_mat);
          const string_mesh = new THREE.Mesh(string_geometry, numb_mat);
          meshs.push({ NUMB_MESH: numb_mesh, STRING_MESH: string_mesh });
          cube.add(numb_mesh, string_mesh);
        }

        const size_mesh = new THREE.Box3().setFromObject(meshs[0].NUMB_MESH); // dimensioni
        var numb_size = new THREE.Vector3();
        var size_number_block = size_mesh.getSize(numb_size);

        const size_mesh_tring = new THREE.Box3().setFromObject(meshs[0].STRING_MESH); // dimensioni
        var string_size = new THREE.Vector3();
        var size_string_block = size_mesh_tring.getSize(string_size);

        // cordinate testo su cubo
        meshs[0].NUMB_MESH.position.set(-(size_number_block.x / 2), -size_number_block.y / 2, 0); // Faccia frontale
        meshs[0].STRING_MESH.position.set(-0.4, -size_number_block.y / 2 + 0.3, 0);

        meshs[1].NUMB_MESH.position.set(size_number_block.x / 2, -size_number_block.y / 2, -0); // Faccia posteriore
        meshs[1].STRING_MESH.position.set(0.4, -size_number_block.y / 2 + 0.3, 0);
        meshs[1].NUMB_MESH.rotation.y = meshs[1].STRING_MESH.rotation.y = Math.PI;

        meshs[2].NUMB_MESH.position.set(0, -size_number_block.y / 2, size_number_block.x / 2); // Faccia destra
        meshs[2].STRING_MESH.position.set(0, -size_number_block.y / 2 + 0.3, size_string_block.x / 2);
        meshs[2].NUMB_MESH.rotation.y = meshs[2].STRING_MESH.rotation.y = Math.PI / 2;

        meshs[3].NUMB_MESH.position.set(0, -size_number_block.y / 2, -size_number_block.x / 2); // Faccia sinistra
        meshs[3].STRING_MESH.position.set(0, -size_number_block.y / 2 + 0.3, -size_string_block.x / 2);
        meshs[3].NUMB_MESH.rotation.y = meshs[3].STRING_MESH.rotation.y = -Math.PI / 2;
      });
      scene.add(cube);
    }

    camera.position.z = 6;
    camera.position.y = 5;
    console.log(scene.children);

    const animation = () => {
      // è il render delle immagini.  This will create a loop that causes the renderer to draw the scene every time the screen is refreshed (on a typical screen this means 60 times per second).
      requestAnimationFrame(animation);
      scene.children.forEach((block, y) => {
        block.rotation.y += 0.01; // rotation or position
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
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    });
  },
};
</script>

<template>
  <block v-if="show_block" :value="blocks" @close="blockchain" />
  <main id="container_blockchain"></main>
  <div v-for="(block, i) in 10" v-if="!show_block">
    <div @click="GetBlock(i)">{{ i }} CICICICICICIICICICICI</div>
  </div>
</template>

<style lang="scss">
@use "./../style/general.scss" as *;
#container_blockchain {
  height: calc(100vh - 90px);
  width: 100%;
}
#block {
  height: 100%;
  width: 100%;
  display: block;
}
</style>
