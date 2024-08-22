<script>
import Block from "./Block.vue";
import axios from "axios";
import * as THREE from "three";
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
        const data1 = await axios("https://blockstream.info/api/block-height/857772"); // blocco specifico
        const data2 = await axios("https://blockstream.info/api/block/" + data1.data); // dati blocco
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
      console.log(typeof this.blocks);
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

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setSize(space.clientWidth, space.clientHeight);
    space.appendChild(renderer.domElement);

    const geometry = new THREE.BoxGeometry(); // dimensioni box
    const material = new THREE.MeshBasicMaterial({ color: "red" });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    camera.position.z = 5;

    const animation = () => {
      requestAnimationFrame(animation);
      cube.rotation.y += 0.01;
      renderer.render(scene, camera);
    };
    animation();

    window.addEventListener("resize", () => {
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

// canvas {
//   width: 100%;
//   height: 100%;
// }
</style>
