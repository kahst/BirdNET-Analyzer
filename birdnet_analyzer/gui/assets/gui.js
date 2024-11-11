function init() {
    function checkForNewerVersion() {
        let gui_version_element = document.getElementById("current-version")
        
        if (gui_version_element && gui_version_element.textContent != "main") {
            console.log("Checking for newer version...");

            function sendGetRequest(url) {
                return new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();
                    xhr.open("GET", url);
                    xhr.onload = () => {
                        if (xhr.status === 200) {
                            resolve(xhr.responseText);
                        } else {
                            reject(new Error(`Request failed with status ${xhr.status}`));
                        }
                    };
                    xhr.onerror = () => {
                        reject(new Error("Request failed"));
                    };
                    xhr.send();
                });
            }

            const apiUrl = "https://api.github.com/repos/kahst/BirdNET-Analyzer/releases/latest";

            sendGetRequest(apiUrl)
                .then(response => {
                    const current_version = document.getElementById("current-version").textContent;
                    const response_object = JSON.parse(response);
                    const latest_version = response_object.tag_name;
                    
                    if (latest_version.startsWith("v")) {
                        latest_version = latest_version.slice(1);
                    }

                    if (current_version !== latest_version) {
                        const updateNotification = document.getElementById("update-available");

                        updateNotification.style.display = "block";
                        const linkElement = updateNotification.getElementsByTagName("a")[0]
                        linkElement.href = response_object.html_url;
                        linkElement.target = "_blank";
                    }
                })
                .catch(error => {
                    console.error(error);
                });
        }
    }

    function overwriteStyles() {
        console.log("Overwriting styles...");
        const styles = document.createElement("style");
        styles.innerHTML = "@media (width <= 1024px) { .app {max-width: initial !important;}}";
        document.head.appendChild(styles);
    }

    checkForNewerVersion();
    overwriteStyles();
}