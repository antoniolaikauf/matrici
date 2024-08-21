<script>
import axios from "axios";
export default {
  name: "Blockchain",
  data() {
    return {};
  },
  methods: {
    async call() {
      try {
        const data = await axios("https://blockstream.info/api/blocks/tip/height");  // altezza ultimo blocco
        const data1 = await axios("https://blockstream.info/api/block-height/88"); // blocco specifico 
        const data2 = await axios("https://blockstream.info/api/block/" + data1.data); // dati blocco
        console.log(data2.data);
        console.log(data1.data);
        console.log(data);
      } catch (error) {
        console.log(error.data);
      }
    },
  },
  mounted() {
    this.call();
  },
};
</script>

<template>
  <div>ciaoaoaoa</div>
</template>

<style scoped lang="sass"></style>
