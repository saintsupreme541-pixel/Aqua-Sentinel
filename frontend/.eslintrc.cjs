module.exports = {
  root: true,
  env: { browser: true, es2021: true },
  parser: "@typescript-eslint/parser",
  parserOptions: { ecmaVersion: "latest", sourceType: "module", ecmaFeatures: { jsx: true } },
  plugins: ["@typescript-eslint", "react-hooks"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react-hooks/recommended",
  ],
  ignorePatterns: ["dist", "node_modules"],
  rules: {
    // @typescript-eslint already covers unused vars for TS files
    "no-unused-vars": "off",
    // React 17+ JSX transform: no need to import React in every file
    "react/react-in-jsx-scope": "off",
  },
};
