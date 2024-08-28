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
    return { timestamp: null };
  },
  mounted() {
  },
};
</script>
<template>
  <div id="block">
    <div>Height: {{ value.height }}</div>
    <div>Timestamp: {{ new Date(this.value.timestamp * 1000).toLocaleString() }}</div>
    <div>Size: {{ value.size }}</div>
    <div>Weight: {{ value.weight }}</div>
    <div>Transazioni: {{ value.tx_count }}</div>
    <div>Merkle root: {{ value.merkle_root }}</div>
    <div>Previous block hash: {{ value.previousblockhash }}</div>
    <button type="button" class="btn btn-outline-light" @click="$emit('close')">Close</button>
  </div>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;
#block {
  width: 80%;
  margin: auto;
  div {
    padding: 10px 0;
    cursor: pointer;
  }
}
</style>
