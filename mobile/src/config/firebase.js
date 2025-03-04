import { initializeApp } from 'firebase/app';

const firebaseConfig = JSON.parse(process.env.FIREBASE_CONFIG);

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);

export default firebaseApp;