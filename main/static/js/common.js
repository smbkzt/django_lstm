$('#ajax-form').on('submit', function(event){
    event.preventDefault();
    $.ajax({
        url : "/",
        type : "POST",
        data : { "the_origin" : $('#input-origin-message').val(),
                 "the_comment": $('#input-comment-message').val() },
        success : function(json) {
            if (json == "The comment message has agreement sentiment"){
                $("#answer").html('<div class="alert alert-success">' + json + '</div>');
            }
            else{
                $("#answer").html('<div class="alert alert-danger">' + json + '</div>');
            }
        },
        error : function(xhr,errmsg,err) {
            $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+
                " <a href='#' class='close'>&times;</a></div>");
            console.log(xhr.status + ": " + xhr.responseText);
        }
    });
});
