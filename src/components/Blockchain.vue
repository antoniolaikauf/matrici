<script>
import Block from "./Block.vue";
import axios from "axios";
import * as THREE from "three";
import { Wireframe } from "three/examples/jsm/Addons.js";
import { ior } from "three/src/nodes/core/PropertyNode.js";
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
    this.call();
    const space = document.getElementById("container_blockchain");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, space.clientWidth / space.clientHeight, 0.1, 1000);

    const spotLight = new THREE.SpotLight(0xffffff, 1.0);
    console.log(spotLight);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setSize(space.clientWidth, space.clientHeight);
    space.appendChild(renderer.domElement);

    let distance = 0;
    const numbblock = 5;
    const group = new THREE.Group();
    for (let i = 0; i < numbblock; i++) {
      const geometry = new THREE.BoxGeometry(); // dimensioni box
      let material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(138 / 255, 138 / 255, 138 / 255), // Grigio acciaio RGB(138, 138, 138)
        metalness: 1, // Alta metallicità
        roughness: 0.3, // Leggera ruvidità
        emissive: new THREE.Color(0, 0, 0), // Nessuna emissione
      });
      console.log(material);

      const cube = new THREE.Mesh(geometry, material);
      cube.position.set(0, distance, 0);
      distance += 2;
      group.add(cube);
    }

    scene.add(group);
    camera.position.z = 6;
    camera.position.y = 5;

    const animation = () => {
      requestAnimationFrame(animation);
      group.children.forEach((block, i) => {
        block.rotation.y += 0.01;
      });
      spotLight.position.set(0, 10, 0); // Posiziona la luce sopra i blocchi
      spotLight.target.position.set(0, 0, 0); // La luce punta verso il centro dei blocchi
      scene.add(spotLight);
      scene.add(spotLight.target); // Aggiungi il target per la luce
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
  // background-color: rgb(107, 93, 93);
}
#block {
  height: 100%;
  width: 100%;
  display: block;
}
</style>
