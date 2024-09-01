import mitt from "mitt"; 
const eventBus = mitt()
export default eventBus
// bus che permette di far ascoltare un evento a tutti i componenti