$('#ajax-lstm-form').on('submit', function(event){
    event.preventDefault();
    $(".loader").css("display", "inline-block");
    $("#answer").css("display", "none");
    $.ajax({
        url : "/try-lstm",
        type : "POST",
        data : {
            "the_origin" : $('#input-origin-message').val(),
            "the_comment": $('#input-comment-message').val()
        },
        success : function(json) {
            $(".loader").css("display", "none");
            $("#answer").css("display", "block");
            if (json == "The comment message has agreement sentiment"){
                $("#answer").html('<div class="alert alert-success">' + json + '</div>');
            }
            else{
                $("#answer").html('<div class="alert alert-danger">' + json + '</div>');
            }
        },
        error: function(data){
            $("#answer").html('<div class="alert alert-danger">' + data + '</div>');
        }
    });
});

$('#train-form').on('submit', function(event){
    $(".loader").css("display", "inline-block");
});

function sleep (time) {
    return new Promise((resolve) => setTimeout(resolve, time));
}

$('#ajax-tweet-form').on('submit', function(event){
    event.preventDefault();
    $(".loader").css("display", "inline-block");
    $("#answer").css("display", "inline-block");
    $.ajax({
        url : "/tweet-search",
        type : "POST",
        data : {
            "keywords" : $('#input-keyword').val()
        },
        success : function(json) {
            var data = JSON.stringify(json);
            $(".loader").css("display", "none");
            $("#answer").css("display", "none");
            var ul = document.getElementById('message-bubbles');
            ul.innerHTML = '';
            $(".wrapper-visible").addClass("animated fadeOutRight");
            $(".wrapper-visible").css("position", "absolute");
            sleep(200).then(() => {
                $(".wrapper-visible").css("display", "none");
            });

            $(".wrapper-hidden").addClass("animated fadeInLeftBig");
            $(".wrapper-hidden").css("display", "inline-block");
            var parsed = JSON.parse(data);
            var container = document.getElementById('mynetwork');
            var nodes = new vis.DataSet(parsed.nodes);
            var edges = new vis.DataSet(parsed.edges);
            var data = {
                nodes: nodes,
                edges: edges
            };
            var options = {
                edges:{
                    arrows: {
                        from: {enabled: true, scaleFactor:0.75, type:'arrow'}
                    },
                    length: 150,
                },
                nodes:{
                    brokenImage: undefined,
                    shape: 'circularImage',
                    chosen: true,
                }
            }
            var network = new vis.Network(container, data, options);

            network.on('click', function(properties){
                try{
                    var id = properties.edges;
                    var clickedEdge = data.edges.get(id);

                    var node1 = data.nodes.get(clickedEdge[0].from);
                    var node2 = data.nodes.get(clickedEdge[0].to);

                    var ul = document.getElementById('message-bubbles');
                    ul.innerHTML = "";
                    var li1 = document.createElement("li");
                    var li2 = document.createElement("li");
                    var spanleft = document.createElement("span")
                    var spanright = document.createElement("span")

                    spanleft.setAttribute("class", "left")
                    spanleft.appendChild(document.createTextNode(node1.title))
                    li1.appendChild(spanleft);
                    ul.appendChild(li1);

                    spanright.setAttribute("class", "right")
                    spanright.appendChild(document.createTextNode(node2.title))
                    li2.appendChild(spanright);
                    ul.appendChild(li2);
                }
                catch(TypeError){
                    console.log("Nothin special. Empty place clicked.")
                }
            });
        },
        error: function(data){
            $("#answer").html('<div class="alert alert-danger">' + data + '</div>');
        }
    });
});
