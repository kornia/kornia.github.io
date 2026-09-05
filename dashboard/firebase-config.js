// Firebase web app configuration for the kornia hub (project kornia-website).
// These values are public identifiers by design (they end up in every visitor's browser); the security
// lives in Firebase Authentication, the Firestore rules (firestore.rules) and the HTTP-referrer
// restriction on the API key in the Google Cloud console.
window.KORNIA_FIREBASE = {
  configured: true,
  apiKey: "AIzaSyAlxiV2F_aIwLoV2rSGF-S0lNbXQXfWAak",
  authDomain: "kornia-website.firebaseapp.com",
  projectId: "kornia-website",
  storageBucket: "kornia-website.firebasestorage.app",
  messagingSenderId: "109468841240",
  appId: "1:109468841240:web:3c9056f8bb8b301ca2da6d",
  measurementId: "G-VHXZ7PLVHS",   // Google Analytics; not initialised by hub.js
  // Two environments, chosen by the host this page is served from, so the same files serve both and a page
  // can never reach the wrong service: production on kornia.org, development on the preview site and localhost.
  // The dashboard's ?api=... override still points a page at a local uvicorn.
  environments: {
    prod: { hosts: ["www.kornia.org", "kornia.org"], apiBase: "https://hub-api-109468841240.europe-west1.run.app", usersRoot: "users" },
    dev:  { hosts: ["shijianjian.github.io", "localhost", "127.0.0.1"], apiBase: "https://hub-api-dev-109468841240.europe-west1.run.app", usersRoot: "dev_users" },
  },
};
(function () {
  var envs = window.KORNIA_FIREBASE.environments, host = location.hostname;
  var name = Object.keys(envs).find(function (k) { return envs[k].hosts.indexOf(host) !== -1; }) || "dev";
  window.KORNIA_FIREBASE.env = name;
  window.KORNIA_FIREBASE.apiBase = envs[name].apiBase;
  window.KORNIA_FIREBASE.usersRoot = envs[name].usersRoot;
})();
