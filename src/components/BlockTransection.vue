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
  <div>{{ value_transection }}</div>
  <button type="button" class="btn btn-outline-light" @click="$emit('close')">Close</button>
</template>

<style scoped lang="scss">
@use "./../style/general.scss" as *;
</style>
