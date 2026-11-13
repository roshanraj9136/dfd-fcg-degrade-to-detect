// Set light or dark before the page paints (no flash): the visitor's saved choice, else the device setting.
(function () {
  try {
    var saved = localStorage.getItem("theme");
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
    if (saved) {
      var metas = document.querySelectorAll('meta[name="theme-color"]');
      for (var i = 0; i < metas.length; i++) metas[i].setAttribute("content", dark ? "#0b1120" : "#4f46e5");
    }
  } catch (e) { /* storage blocked: stay light */ }
})();
