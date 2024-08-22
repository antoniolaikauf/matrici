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
  <div>
    {{ value }}
  </div>
  <button @click="$emit('close')">chiudi</button>
</template>

<style  lang="scss">
@use "./../style/general.scss" as *;
</style>
