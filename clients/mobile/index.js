import { registerRootComponent } from 'expo';
import App from './App';

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that you don't need to call AppRegistry.runApplication in an Android project.
registerRootComponent(App);