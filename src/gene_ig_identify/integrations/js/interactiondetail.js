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
let fs = require('fs/promises');

//let utils = require('./utils.js');

let myArgs = process.argv.slice(2);
if(myArgs.length != 4) {
    console.log("Usage: node interactiondetail.js [PDB ID] [Chain 1] [Chain 2] [Output file name]");
    return;
}

let inputid = myArgs[0].toUpperCase(); //'6jxr'; //myArgs[0];
let chain1 = myArgs[1];
let chain2 = myArgs[2];
let outputFileName = myArgs[3];

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
    
      
        let chainid1 = inputid + '_' + chain1;
        let chainid2 = inputid + '_' + chain2;
        if(!ic.chains.hasOwnProperty(chainid1) || !ic.chains.hasOwnProperty(chainid2)) {
            console.error("Error: The chain " + chainid1 + " or " + chainid2 + " does not exit...");
            return;
        }

        // prepare names for two sets
        let residueHash1 = ic.firstAtomObjCls.getResiduesFromAtoms(ic.chains[chainid1]);
        let command1 = chainid1;
        let residArray1 = Object.keys(residueHash1);
        ic.selectionCls.addCustomSelection(residArray1, command1, command1, 'select ' + command1, true);
        let nameArray1 = [command1];

        let residueHash2 = ic.firstAtomObjCls.getResiduesFromAtoms(ic.chains[chainid2]);
        let command2 = chainid2;
        let residArray2 = Object.keys(residueHash2);
        ic.selectionCls.addCustomSelection(residArray2, command2, command2, 'select ' + command2, true);
        let nameArray2 = [command2];

        let type = 'save1';
        let result = await ic.viewInterPairsCls.viewInteractionPairs(nameArray1, nameArray2, false, type,
              true, true, true, true, true, true);
        const jsonData = JSON.stringify(result, null, 2);
        await fs.writeFile(outputFileName, jsonData, { encoding: 'utf8' });

    });
}).on('error', function(e) {
    console.error("Error: " + inputid + " has no MMDB data...");
});
