// Headless Script testing
// Reusable slimerjs/phantomjs script for running cljs.test tests

var p = require('webpage').create();
var fs = require('fs');
var sys = require('system');

var pagePath = fs.absolute("doo-index.html");

var scripts = "";

for (var i = 1; i < sys.args.length; i++) {
    scripts += '<script src="' + fs.absolute(sys.args[i]) + '"></script>';
}

var html = "<html><head><meta charset=\"UTF-8\">"
           + scripts
           + "</head><body></body></html>";

fs.write(pagePath, html, 'w');

function isSlimer() {
    return (typeof slimer !== 'undefined');
}

function exit(code) {
    if (isSlimer()) {
        slimer.exit(code);
    } else {
        phantom.exit(code);
    }
}

// Prepare the page with triggers

// Redirects the output of the page into the script
p.onConsoleMessage = function(msg) {
    console.log(msg);
};

p.onError = function(msg) {
    console.error(msg);
    exit(1);
};

p.open("file://" + pagePath, function (status) {
    fs.remove(pagePath);
    if (status == "fail") {
        console.log(noScriptMsg(isSlimer() ? "Slimer" : "Phantom"));
        exit(1);
    } else {

        p.onCallback = function (x) {
	    var line = x.toString();
	    if (line !== "[NEWLINE]") {
	        console.log(line.replace(/\[NEWLINE\]/g, "\n"));
	    }
        };

        // p.evaluate is sandboxed, can't ship closures across;
        // so, a bit of a hack, better than polling :-P
        var exitCodePrefix = "phantom-exit-code:";
        p.onAlert = function (msg) {
	    var exitCode = msg.replace(exitCodePrefix, "");
	    if (msg != exitCode) {
                exit(parseInt(exitCode));
            }
        };

        // if slimer & async add a shim to avoid insecure operations
        if (isSlimer()) {
            p.evaluate(function () {
                function isAsync() {
                    return ((typeof goog !== 'undefined') &&
                            (typeof goog.async !== 'undefined'));
                }
                if (isAsync()) {
                    goog.async.nextTick.setImmediate_ = function(fnCall) {
                        return window.setTimeout(fnCall, 0);
                    };
                }
            });
        }

        p.evaluate(function(hasDoo, noDooMsg, exitCodePrefix) {

            // Helper funciton to exit
            function exit(isSuccess) {
                window.alert(exitCodePrefix + (isSuccess ? 0 : 1));
	    }
            if (hasDoo()) {
                // If doo is present, set up printing and exit, and start!
	        doo.runner.set_print_fn_BANG_(function(x) {
	            // using callPhantom to work around
                    // https://github.com/laurentj/slimerjs/issues/223
	            window.callPhantom(x.replace(/\n/g, "[NEWLINE]"));
                    // since console.log *itself* adds a newline
	        });
	        doo.runner.set_exit_point_BANG_(exit);
                // Start the tests, which should call the exit
                // function when they are done.
                var results = doo.runner.run_BANG_();

            } else {
                // If doo is not present, present an error and exit
                window.callPhantom(noDooMsg);
                exit(false);
            }
        }, hasDoo, noDooMsg(), exitCodePrefix, sys.env["CATALOG_URL"], sys.env["API_TOKEN"]);
    }
});
// Common code to all runners
// Should be kept in sync between rhino.js & headless.js

function hasDoo () {
    return (typeof doo !== "undefined" &&
            typeof doo.runner !== "undefined" &&
            typeof doo.runner.run_BANG_ === "function");
};

function noDooMsg() {
    var msgLines = [
            "",
            "ERROR: doo was not loaded from the compiled script.",
            "",
            "Make sure you start your tests using doo-tests or doo-all-tests",
            "and that you include that file in your build",
            ""
        ];
    return msgLines.join("\n");
};

function noScriptMsg(env) {
    var msgLines = [
        "",
        ["ERROR:", env, "couldn't find the script"].join(" "),
        "",
        "This is most likely doo's fault. Please file an issue at ",
        "http://github.com/bensu/doo and we'll sort it out together.",
        ""
    ];
    return msgLines.join("\n");
}
