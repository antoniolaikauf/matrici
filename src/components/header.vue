<script>
import eventBus from "../eventBus";
import axios from "axios";
export default {
  name: "header",
  data() {
    return { transection: "" };
  },
  methods: {
    async transection_information() {
      const info_transection = await axios(`https://api.blockcypher.com/v1/btc/main/txs/${this.transection}`);
      eventBus.emit("New_block", { msg: this.transection, info_block: info_transection.data, value: true });
      console.log(info_transection.data);
    },
  },
};
</script>
<template>
  <header>
    <nav class="navbar">
      <img src="../../public/img/logo.png" alt="" />
      <div class="input-group mb-3" id="search">
        <button @click="transection_information" class="btn btn-outline-secondary" type="button" id="button-addon1">Search</button>
        <input
          v-model="transection"
          type="text"
          class="form-control"
          placeholder="transection"
          aria-label="Example text with button addon"
          aria-describedby="button-addon1"
        />
      </div>
    </nav>
  </header>
</template>
<style lang="scss" scoped>
@use "./../style/general.scss" as *;

header {
  height: 90px;
  .navbar {
    display: flex;
    position: fixed;
    top: 0px;
    width: 100%;
    justify-content: space-between;
    padding: 10px 30px;
    #search {
      width: 30%;
    }
    img {
      width: 50px;
      height: 50px;
    }
  }
}
</style>
