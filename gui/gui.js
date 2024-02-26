function checkForNewerVersion() {
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
            const current_version = "v" + document.getElementById("current-version").textContent;
            const response_object = JSON.parse(response);
            const latest_version = response_object.tag_name;

            if (current_version !== latest_version) {
                const updateNotification = document.getElementById("update-available");

                updateNotification.style.display = "block";
                updateNotification.getElementsByTagName("a")[0].href = response_object.html_url;
            }
        })
        .catch(error => {
            console.error(error);
        });
}