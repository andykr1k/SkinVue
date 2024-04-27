/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        turquoise: "#35D4C2",
        lightred: "#FF7070",
        richblack: "#011627",
        ghostwhite: "#F8F4F9",
        dimgrey: "#726E75",
      },
    },
  },
  plugins: [],
};
