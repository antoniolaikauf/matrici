<script>
import axios from "axios";
import * as THREE from "three";
export default {
  name: "Blockchain",
  data() {
    return {
      blocks: "",
    };
  },
  methods: {
    async call() {
      try {
        const data = await axios("https://blockstream.info/api/blocks/tip/height"); // altezza ultimo blocco
        const data1 = await axios("https://blockstream.info/api/block-height/857772"); // blocco specifico
        const data2 = await axios("https://blockstream.info/api/block/" + data1.data); // dati blocco
        this.blocks = data.data;
        console.log(data2.data);
        console.log(data1.data);
        console.log(data);
      } catch (error) {
        console.log(error.data);
      }
    },
    async GetBlock(i) {
      const blockHash = await axios(`https://blockstream.info/api/block-height/${i}`);
      console.log(blockHash.data);
      const dataBlock = await axios(`https://blockstream.info/api/block/${blockHash.data}`); // dati blocco
        console.log(dataBlock.data);
    // difficolta è quanto è difficile minare un blocco, un aumento del target diminuisce la difficolta, una diminuzione del target aumenta la difficolta 
    },
  },
  mounted() {
    this.call();
  },
};
</script>

<template>
  <div v-for="(block, i) in 10" @click="GetBlock(i)">ciaoao</div>
  <div>ciaoaoaoa</div>
</template>

<style scoped lang="sass"></style>
