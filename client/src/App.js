import React from 'react';

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      imageURL: '',
      predict:null
    };

    this.handleUploadImage = this.handleUploadImage.bind(this);
  }

  handleUploadImage(ev) {
    console.log("FUNCTION CLLED")
    ev.preventDefault();
    var myHeaders = new Headers();
    myHeaders.append("Content-Type", "multipart/form-data");
    let data = new FormData();
    data.append("file", this.uploadInput.files[0]);
    data.append("target", this.fileName.value);
    // data.append('file', this.uploadInput.files[0]);
    // data.append('filename', this.fileName.value);
    console.log(data,"file", this.uploadInput.files[0])
    fetch('http://127.0.0.1:5000/upload', {
      method: 'POST',
      headers: myHeaders,
      body: data,
      redirect: 'follow'
    }).then((response) => {
      console.log("EEE")
      response.json().then((body) => {
        console.log(body)
        console.log("HELLEW");
        this.setState({predict:body})
        // this.setState({ imageURL: `http://localhost:8000/${body.file}` });
      });
    }).catch((err) => {
      console.log(err,'error')
    })
  }

  render() {
    return (
      <form onSubmit={this.handleUploadImage}>
        <div>
          <input ref={(ref) => { this.uploadInput = ref; }} type="file" />
        </div>
        <div>
          <input ref={(ref) => { this.fileName = ref; }} type="text" placeholder="Enter the desired name of file" />
        </div>
        <br />
        <div>
          <button>Upload</button>
        </div>
        {this.state.predict?
        <div>this.state.predict</div>
        :
        null
        }
      </form>
    );
  }
}

export default App;