/* HTML Elments*/
const actualBtn = document.getElementById('folder-upload-btn');
const fileChosen = document.getElementById('files-uploaded');
const table = document.getElementById('file-table');


/* File sizes */
const units = ['bytes', 'kB', 'MB', 'GB', 'TB'];

actualBtn.addEventListener('change', function(){
  fileChosen.textContent = this.files.length + " files";
  let file, fsize, funit;
  let rows = '<tr><th>File Name</th><th>File Size</th></tr>';

  for(let i=0;i<this.files.length;i++){
    file = this.files[i];
    fsize = file.size;
    
    // fixing file size and file unit
    let unitIndex = 0;
    while(fsize >= 1024 && unitIndex < 4){
      fsize /= 1024
      unitIndex++;
    }
    funit = units[unitIndex];

    // adding new row 
    rows += '<tr><td>' + file.name + '</td>' + '<td>' +  fsize.toFixed(1) + ' ' + funit + '</td>' +  '</tr>';
  }

  // adding rows to table
  table.innerHTML = table.innerHTML + rows;

  document.getElementById('end-note').style.visibility = 'hidden';
})