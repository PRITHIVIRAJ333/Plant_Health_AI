async function predictHealth() {

    const soilMoisture =
        document.getElementById(
            "soil_moisture"
        ).value;

    const temperature =
        document.getElementById(
            "temperature"
        ).value;

    const humidity =
        document.getElementById(
            "humidity"
        ).value;

    const leafMoisture =
        document.getElementById(
            "leaf_moisture"
        ).value;

    const sunlight =
        document.getElementById(
            "sunlight"
        ).value;

    const soilPH =
        document.getElementById(
            "soil_ph"
        ).value;


    if (
        soilMoisture === "" ||
        temperature === "" ||
        humidity === "" ||
        leafMoisture === "" ||
        sunlight === "" ||
        soilPH === ""
    ) {

        alert(
            "Please enter all values."
        );

        return;
    }


    document
        .getElementById("loading")
        .classList
        .remove("hidden");


    document
        .getElementById("result")
        .classList
        .add("hidden");


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        soil_moisture:
                            soilMoisture,

                        temperature:
                            temperature,

                        humidity:
                            humidity,

                        leaf_moisture:
                            leafMoisture,

                        sunlight:
                            sunlight,

                        soil_ph:
                            soilPH
                    })
                }
            );


        const data =
            await response.json();


        document
            .getElementById("loading")
            .classList
            .add("hidden");


        if (data.success) {

            document
                .getElementById("result")
                .classList
                .remove("hidden");


            document
                .getElementById("result-title")
                .innerText =
                data.result
                    .replace(
                        "_",
                        " "
                    )
                    .toUpperCase();


            document
                .getElementById("confidence")
                .innerText =
                data.confidence;


            document
                .getElementById("message")
                .innerText =
                data.message;

        } else {

            alert(data.message);

        }


    } catch (error) {

        document
            .getElementById("loading")
            .classList
            .add("hidden");

        alert(
            "Server connection error."
        );

        console.error(error);
    }
}