# kornia.org/hub

The account side of the site, on Google's stack: Firebase Authentication (GitHub, Google, email and
password), Firestore for per-user data, and later Cloud Run for the API and the GPU-backed heavy models.
The front end is static and lives here; nothing in this folder is a secret.

## What the page does today

- Sign in with GitHub or Google, or register with an email address (verification mail, password reset).
- Shows the profile and lets the user sign out.
- Syncs the playground's saved pipelines (browser storage) to `users/{uid}/pipelines` and lists them
  with "open in the playground" share links, so a user's work follows them across devices.
- Shows the account's server usage from `hub-api/`: today's quota, 30 days of runs, the recent runs, the tier, and a delete-account button. The server models themselves live in the playground's
  Models section, marked "server"; pressing Run there without an account opens a sign-in prompt.
- `auth.js` at the site root renders the header's Sign in / avatar control on every page and exposes
  `window.KorniaAuth` (user, token, sign-in, sign-out) to the playground. Opening the hub with `?next=/path`
  returns the visitor to that path after signing in.

## Setting it up (one-time, in the Firebase console)

1. Create a Firebase project inside the Google Cloud project you will use for everything else. Pick a
   European default location (`europe-west1` or `europe-west3`) for Firestore; it cannot be changed later.
2. Authentication > Sign-in method: enable **Email/Password** (with email verification), **Google**, and
   **GitHub**. For GitHub, create an OAuth app at github.com/settings/developers with the callback URL the
   console shows (`https://<project>.firebaseapp.com/__/auth/handler`) and paste its client id and secret.
3. Authentication > Settings > Authorized domains: add `www.kornia.org`, `kornia.org` and the preview host
   `shijianjian.github.io`.
4. Firestore: create the database in production mode and publish `firestore.rules` from this folder
   (`firebase deploy --only firestore:rules` with the CLI, or paste it in the Rules tab).
5. Project settings > Your apps > add a Web app; copy its config into `firebase-config.js` and set
   `configured: true`. The API key there is a public identifier, restrict it to the site's domains in the
   Google Cloud console under APIs & Services > Credentials.
6. Authentication > Templates: set the sender name and a kornia.org reply address; for volume, connect a
   custom SMTP (Authentication > Templates > SMTP settings).

Until step 5 is done the page shows these instructions instead of the sign-in buttons.

## The server side

`../hub-api/` is the FastAPI service for Cloud Run: token verification, a daily quota per user in
Firestore, and the models (DexiNed edges and LoFTR matching to start, both kornia's own, fast enough on
CPU). Its README has the deploy command; afterwards put the service URL into `firebase-config.js` as
`apiBase`. For a local test, `?api=http://127.0.0.1:8090` on the hub URL points the page at a local
server and lets an unverified test account through.

## Next pieces

- GPU models on Cloud Run (Segment Anything, ControlNet from kornia edges, Depth Anything large, a
  pipeline on a whole video), a job queue once runs exceed a request's lifetime.
- Sponsor tiers from Open Collective, mapped to the `sponsor` custom claim the API already honours.
- Public galleries: an explicitly published collection with its own read-only rule.
