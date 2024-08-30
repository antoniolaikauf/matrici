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
  <div class="block">
    <div class="text-center">
      <h5>Transazione</h5>
      <span>{{ value_transection.hash }}</span>
    </div>
    <div>Transazione nel blocco {{ value_transection.block_height }}</div>
    <div>Transazione confermata {{ value_transection.confirmed }}</div>
    <div>Transazione inviata a: {{ value_transection.outputs[0].addresses[0] }}</div>
    <div>Transazione inviata da: {{ value_transection.outputs[0].spent_by }}</div>
    <div>Quantità inviata {{ Math.round(value_transection.total / 100000000).toFixed(2) }}</div>
    <button type="button" class="btn btn-outline-light" @click="$emit('close')">Close</button>
  </div>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;
@use "./../style/__mixins.scss" as *;
</style>
