<script>
export default {
  name: "transections_block",
  props: {
    value_transection: {
      type: Object,
    },
  },
};
</script>

<template>
  <div id="block">
    <div>Transazione nel blocco {{ value_transection.block_height }}</div>
    <div>Transazione confermata {{ value_transection.confirmed }}</div>
    <div>Transazione inviata a {{ value_transection.outputs[0].addresses[0] }}</div>
    <div>Transazione inviata da {{ value_transection.outputs[0].spent_by }}</div>
    <div>Quantità inviata {{ Math.round(value_transection.total / 100000000).toFixed(2) }}</div>
    <button type="button" class="btn btn-outline-light" @click="$emit('close')">Close</button>
  </div>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;

#block {
  width: 80%;
  margin: auto;
  div {
    padding: 10p 0;
    cursor: pointer;
  }
}
</style>
