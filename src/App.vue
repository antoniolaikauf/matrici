<script>
import Blockchain from "./components/Blockchain.vue";
import header_page from "./components/header.vue";
export default {
  components: {
    Blockchain,
    header_page,
  },
};
</script>

<template>
  <header_page> </header_page>
  <Blockchain />
</template>

<style lang="scss">
@use "./style/general.scss" as *;
// body {
//   background: url("../public/img/star.jpg");
//   // background-size: cover;
// }
</style>
