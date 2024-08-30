<script>
import eventBus from "./eventBus";
import { defineAsyncComponent } from "vue";
import axios from "axios";
import * as THREE from "three";
// non fanno più parte della libreria bisogna importarli manualemnte
import { FontLoader } from "three/examples/jsm/loaders/FontLoader.js";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";
export default {
  name: "Blockchain",
  components: {
    transections_block: defineAsyncComponent(() => import("./BlockTransection.vue")),
    block: defineAsyncComponent(() => import("./Block.vue")), // lazy loader viene caricata solo quando si ha bisogno del componente
  },
  data() {
    return {
      blocks: [],
      show_block: false,
      block_iesimo: "",
      T_block: "",
      T_show: false,
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
      const dataBlock = await axios(`https://api.blockcypher.com/v1/btc/main/blocks/${i}?txstart=1&limit=10`);
      this.block_iesimo = dataBlock.data;
    },
    blockchain() {
      this.show_block = false;
      this.T_show = false;
    },
    size(ob) {
      const size_mesh = new THREE.Box3().setFromObject(ob);
      const mesh_size = new THREE.Vector3();
      const size_block = size_mesh.getSize(mesh_size);
      return size_block;
    },
    posizion_text(texts, n, s) {
      texts[0].NUMB_MESH.position.set(-(n.x / 2), -n.y / 2, 0.05); // Faccia frontale
      texts[0].STRING_MESH.position.set(-0.4, -n.y / 2 + 0.3, 0.05);

      texts[1].NUMB_MESH.position.set(n.x / 2, -n.y / 2, -0.05); // Faccia posteriore
      texts[1].STRING_MESH.position.set(0.4, -n.y / 2 + 0.3, -0.05);
      texts[1].NUMB_MESH.rotation.y = texts[1].STRING_MESH.rotation.y = Math.PI;

      texts[2].NUMB_MESH.position.set(0.05, -n.y / 2, n.x / 2); // Faccia destra
      texts[2].STRING_MESH.position.set(0.05, -n.y / 2 + 0.3, s.x / 2);
      texts[2].NUMB_MESH.rotation.y = texts[2].STRING_MESH.rotation.y = Math.PI / 2;

      texts[3].NUMB_MESH.position.set(-0.05, -n.y / 2, -n.x / 2); // Faccia sinistra
      texts[3].STRING_MESH.position.set(-0.05, -n.y / 2 + 0.3, -s.x / 2);
      texts[3].NUMB_MESH.rotation.y = texts[3].STRING_MESH.rotation.y = -Math.PI / 2;
    },
  },
  async mounted() {
    eventBus.on("increment", (data) => {
      // evento bus ascolto
      if (data.value === true) (this.T_show = true), (this.T_block = data.info_block);
    });

    const token = "d4a50872e7484dbeb7550a4a00a11839";
    const new_block = new WebSocket(`wss://socket.blockcypher.com/v1/btc/main?token=${token}`);
    new_block.onopen = () => {
      console.log("Connected to BlockCypher WebSocket server.");
      new_block.send(
        JSON.stringify({
          event: "new-block", // Subscribe to block events
        })
      );
    };
    new_block.onmessage = (event) => {
      var tx = JSON.parse(event.data);
      this.blocks.unshift(tx);
      this.blocks.pop();
      console.log(tx);
      console.log(this.blocks);
    };

    new_block.onclose = () => {
      console.log("close connection");
    };

    await this.call();

    const space = document.getElementById("container_blockchain");

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
      scene.add(cube.clone()); // add block to scena
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

        const size_number = this.size(meshs[0].NUMB_MESH);
        const size_string = this.size(meshs[0].STRING_MESH);

        this.posizion_text(meshs, size_number, size_string);
        // cordinate testo su cubo
      });
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
  },
  beforeUnmount() {
    eventBus.off("increment");
  },
};
</script>

<template>
  <main>
    <transections_block v-if="T_show" :value_transection="T_block" @close="blockchain" />
    <block v-show="show_block" :value="block_iesimo" @close="blockchain" />
    <div class="container-fluid" v-show="!show_block && !T_show">
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
