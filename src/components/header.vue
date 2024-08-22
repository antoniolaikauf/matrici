<script>
export default {
  name: "header",
};
</script>
<template>
  <nav class="navbar">
    <div class="container_nav">
      <img src="../../public/img/logo.png" alt="" />
    </div>
  </nav>
</template>
<style lang="scss" scoped>
@use "./../style/general.scss" as *;

.navbar {
  position: fixed;
  top: 0px;
  width: 100%;

  .container_nav {
    display: flex;
    justify-content: space-between;
    padding: 10px 30px;
    img {
      width: 50px;
      height: 50px;
    }
  }
}
</style>
