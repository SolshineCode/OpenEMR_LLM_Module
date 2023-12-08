<!DOCTYPE html>
<html>
<head>
  <title>Medical Assistant LLM</title>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script>
    $(document).ready(function(){
      $("#submit").click(function(){
        var prompt = $("#prompt").val();
        // Fetch patient data if available
        var patient_data = ""; // Implement logic to retrieve patient data
        $.ajax({
          url: "http://localhost:5000/generate",  // Replace with your Flask server URL
          type: "post",
          data: JSON.stringify({prompt: prompt, patient_data: patient_data}),
          contentType: "application/json",
          success: function(data){
            $("#response").html(data.response);
          },
          error: function(){
            alert("Error: Could not generate response.");
          }
        });
      });
      $("#feedback").click(function(){
        var feedback = $("#feedback_text").val();
        // Implement logic to send feedback to server
        alert("Feedback submitted. Thank you!");
      });
    });
  </script>
</head>
<body>
  <h1>Medical Assistant LLM</h1>
  <textarea id="prompt" placeholder="Enter your prompt here"></textarea>
  <button id="submit">Submit</button>
  <hr>
  <h3>Response</h3>
  <div id="response"></div>
  <hr>
  <h3>Feedback</h3>
  <textarea id="feedback_text" placeholder="Provide feedback on the response"></textarea>
  <button id="feedback">Submit Feedback</button>
</body>
</html>
