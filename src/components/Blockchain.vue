<script>
import Block from "./Block.vue";
import axios from "axios";
import * as THREE from "three";
// non fanno più parte della libreria bisogna importarli manualemnte
import { FontLoader } from "three/examples/jsm/loaders/FontLoader.js";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";
export default {
  name: "Blockchain",
  components: {
    Block,
  },
  data() {
    return {
      blocks: "",
      show_block: true,
    };
  },
  methods: {
    async call() {
      try {
        const data = await axios("https://blockstream.info/api/blocks/tip/height"); // altezza ultimo blocco
        // this.blocks = data.data;
        console.log(data);
      } catch (error) {
        console.log(error.data);
      }
    },
    async GetBlock(i) {
      this.show_block = false;
      const blockHash = await axios(`https://blockstream.info/api/block-height/${i}`);
      this.blocks = blockHash.data;
      const dataBlock = await axios(`https://blockstream.info/api/block/${blockHash.data}`); // dati blocco
      console.log(dataBlock.data);
      // difficolta è quanto è difficile minare un blocco, un aumento del target diminuisce la difficolta, una diminuzione del target aumenta la difficolta
      // bits è la forma compatta del target
      // size dimensione del blocco in bytes
      // transection count quantita di transazioni
    },
    blockchain() {
      this.show_block = true;
    },
  },
  mounted() {
    // * componenti importanti
    this.call();
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
    const numbblock = 5;
    const group = new THREE.Group();
    for (let i = 0; i < numbblock; i++) {
      const geometry = new THREE.BoxGeometry(1.3, 1.3, 1.3); // dimensioni block
      let material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x8a8a8a),
        metalness: 1, //  metallicità
        roughness: 0.3, //  ruvidità
        emissive: new THREE.Color(0, 0, 0),
      });

      const cube = new THREE.Mesh(geometry, material); // prende una geometria e l'applica al materiale
      cube.position.set(0, distance, 0);
      distance += 2;

      const loader = new FontLoader();
      loader.load("../../node_modules/three/examples/fonts/gentilis_bold.typeface.json", (font) => {
        const geometry_number = new TextGeometry("858045", {
          font: font,
          size: 0.2,
          depth: 0.7, // inizia dal centro del cubo
        });
        const txt_mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(0xff0000),
          metalness: 1, //  metallicità
          roughness: 0.3, //  ruvidità
          emissive: new THREE.Color(0, 0, 0),
        });
        const txt_mesh = new THREE.Mesh(geometry_number, txt_mat);
        const size_mesh = new THREE.Box3().setFromObject(txt_mesh);
        var size = new THREE.Vector3();
        var size_number_block = size_mesh.getSize(size);

        txt_mesh.position.x = -(size_number_block.x / 2);
        txt_mesh.position.y = -(size_number_block.y / 2);

        const geometry_string = new TextGeometry("Number Block", {
          font: font,
          size: 0.1,
          depth: 0.7, // inizia dal centro del cubo
        });
        const string_mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(0xff0000),
          metalness: 1,
          roughness: 0.3,
          emissive: new THREE.Color(0, 0, 0),
        });
        const string_mesh = new THREE.Mesh(geometry_string, string_mat);
        
        string_mesh.position.x = -0.4;
        string_mesh.position.y = txt_mesh.position.y + 0.3;
        cube.add(string_mesh);
        cube.add(txt_mesh);
      });
      group.add(cube);
    }

    scene.add(group);
    camera.position.z = 6;
    camera.position.y = 5;

    const animation = () => {
      // è il render delle immagini.  This will create a loop that causes the renderer to draw the scene every time the screen is refreshed (on a typical screen this means 60 times per second).
      requestAnimationFrame(animation);
      group.children.forEach((block, i) => {
        block.rotation.y += 0.01; // rotation or position
      });
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
  <main id="container_blockchain"></main>
  <!-- <div v-for="(block, i) in 10" v-if="show_block">
    <div @click="GetBlock(i)">{{ i }}</div>
  </div>

  <div id="block">
    <canvas></canvas>
  </div>
  <Block :value="blocks" v-if="!show_block" @close="blockchain"></Block> -->
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
