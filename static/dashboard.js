document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const loading = document.getElementById("loadingScreen");

    if (form) {

        form.addEventListener("submit", function () {

            loading.style.display = "flex";

            // Hide after 1 second
            setTimeout(function () {
                loading.style.display = "none";
            }, 1000);

        });

    }

});