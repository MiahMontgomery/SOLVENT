module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        solvent: {
          black: "#121212",
          neon: "#DFFF00",
          steel: "#c9d1d9",
          chrome: "#f5f7ff",
          haze: "#1b1b1b",
        },
      },
      boxShadow: {
        chrome: "0 0 40px rgba(160, 190, 255, 0.25)",
        glow: "0 0 24px rgba(223, 255, 0, 0.25)",
      },
      keyframes: {
        inflate: {
          "0%": { transform: "scale(0.92)" },
          "50%": { transform: "scale(1.05)" },
          "100%": { transform: "scale(1.0)" },
        },
        pop: {
          "0%": { transform: "scale(1)", opacity: "1" },
          "100%": { transform: "scale(1.25)", opacity: "0" },
        },
        floaty: {
          "0%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-6px)" },
          "100%": { transform: "translateY(0px)" },
        },
        codepulse: {
          "0%": { opacity: "0.25", transform: "translateY(10px)" },
          "50%": { opacity: "0.8" },
          "100%": { opacity: "0.1", transform: "translateY(-10px)" },
        },
      },
      animation: {
        inflate: "inflate 1.6s ease-in-out infinite",
        pop: "pop 0.6s ease-out forwards",
        floaty: "floaty 6s ease-in-out infinite",
        codepulse: "codepulse 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
