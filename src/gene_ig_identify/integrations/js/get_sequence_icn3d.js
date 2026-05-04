// Include the interaction in the same chain
// usage: node interaction.js 1TOP A 10 V

/*
Please install the following three packages in your directory with the file interaction.js
npm install three
npm install jquery
npm install icn3d

npm install axios
npm install querystring
*/

// https://github.com/Jam3/three-buffer-vertex-data/issues/2
global.THREE = require('three');
let jsdom = require('jsdom');
global.$ = require('jquery')(new jsdom.JSDOM().window);

let icn3d = require('icn3d');
let me = new icn3d.iCn3DUI({});

let https = require('https');
let axios = require('axios');
let qs = require('querystring');

//let utils = require('./utils.js');

let myArgs = process.argv.slice(2);
if(myArgs.length != 1) {
    console.log("Usage: node allmissingresidues.js [PDB ID]");
    return;
}

let inputid = myArgs[0].toUpperCase(); //'6jxr'; //myArgs[0];

let AFUniprotVersion = 'v4';

let url = (inputid.length == 4) ? "https://www.ncbi.nlm.nih.gov/Structure/mmdb/mmdb_strview.cgi?v=2&program=icn3d&b=1&s=1&ft=1&bu=0&complexity=2&uid=" + inputid
    : "https://alphafold.ebi.ac.uk/files/AF-" + inputid + "-F1-model_" + AFUniprotVersion + ".pdb";

https.get(url, function(res1) {
    let response1 = [];
    res1.on('data', function (chunk) {
        response1.push(chunk);
    });

    res1.on('end', async function(){
      let dataStr = response1.join('');
      // let dataJson = JSON.parse(dataStr);

      me.setIcn3d();
      let ic = me.icn3d;

      ic.bRender = false;
      

      if(isNaN(inputid) && inputid.length > 5) {
        let header = 'HEADER                                                        ' + inputid + '\n';
        dataStr = header + dataStr;
        await ic.opmParserCls.parseAtomData(dataStr, inputid, undefined, 'pdb', undefined);
      }
      else {
        let dataJson = JSON.parse(dataStr);
        await ic.mmdbParserCls.parseMmdbData(dataJson);
      }

      // const ss = me.htmlCls.setHtmlCls.exportSecondary();
      // console.log(ss);
      // console.dir(ic.chainsSeq, { depth: null, maxArrayLength: null });
      console.log(JSON.stringify(ic.chainsSeq));
    });
}).on('error', function(e) {
    console.error("Error: " + pdbid + " has no MMDB data...");
});
