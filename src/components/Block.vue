<script>
export default {
  name: "Block",
  props: {
    value: {
      type: String,
      required: true,
    },
  },
  data() {
    return {};
  },
  mounted() {},
};
</script>
<template>
  <div id="block">
    <div>Height: {{ value.height }}</div>
    <div>Timestamp: {{ value.timestamp }}</div>
    <div>Size: {{ value.size }}</div>
    <div>Weight: {{ value.weight }}</div>
    <div>Merkle root: {{ value.merkle_root }}</div>
    <div>Previous block hash: {{ value.previousblockhash }}</div>
    <button @click="$emit('close')">chiudi</button>
  </div>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;
#block {
  width: 80%;
  margin: auto;
  div{
    padding: 10px 0;
  }
}
</style>
