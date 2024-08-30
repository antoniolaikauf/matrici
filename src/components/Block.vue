<script>
export default {
  name: "Block",
  props: {
    value: {
      type: Object,
    },
  },
};
</script>
<template>
  <div class="block">
    <div>Height: {{ value.height }}</div>
    <div>Timestamp: {{ value.time }}</div>
    <div>Size: {{ value.size }}</div>
    <div>Transazioni: {{ value.n_tx }}</div>
    <div>Merkle root: {{ value.mrkl_root }}</div>
    <div>Previous block hash: {{ value.prev_block }}</div>
    <button type="button" class="btn btn-outline-light" @click="$emit('close')">Close</button>
  </div>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;
@use "./../style/__mixins.scss" as *;
</style>
