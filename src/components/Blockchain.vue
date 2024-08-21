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
        console.log(data2.data);
        console.log(data1.data);
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

      console.log(blockHash.data);
      const dataBlock = await axios(`https://blockstream.info/api/block/${blockHash.data}`); // dati blocco
      console.log(dataBlock.data);
      // difficolta è quanto è difficile minare un blocco, un aumento del target diminuisce la difficolta, una diminuzione del target aumenta la difficolta
      // bits è la forma compatta del target
      // size dimensione del blocco in bytes
      // transection count quantita di transazioni
    },
      blockchain() {
        console.log('ciicicic');
        
      this.show_block = true;
    },
  },
  mounted() {
    this.call();
  },
};
</script>

<template>
  <div v-for="(block, i) in 10" v-if="show_block">
    <div @click="GetBlock(i)">{{ i }}</div>
  </div>
  <Block :value="blocks" v-if="!show_block" @close="blockchain"></Block>
</template>

<style scoped lang="sass"></style>
