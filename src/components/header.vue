<script>
export default {
  name: "header",
};
</script>
<template>
  <header>
    <nav class="navbar">
      <div class="container_nav">
        <img src="../../public/img/logo.png" alt="" />
      </div>
    </nav>
  </header>
</template>
<style lang="scss" scoped>
@use "./../style/general.scss" as *;

header {
  height: 90px;
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
}
</style>
