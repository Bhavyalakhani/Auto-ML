import React from 'react';
import axios from 'axios';
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
    const data = new FormData();
    console.log(this.uploadInput.files[0].name)
    data.append("file", this.uploadInput.files[0]);
    data.append("target", this.fileName.value);
    console.log(this.fileName.value)
    // data.append('file', this.uploadInput.files[0]);
    // data.append('filename', this.fileName.value);
    console.log(data, this.uploadInput.files[0])
    // fetch('http://127.0.0.1:5000/upload', {
    //   method: 'POST',
    //   headers: {
    //     "Content-Type":"multipart/form-data"
    //   },
    //   body: data
    // })
    // .then((response) =>response.text())
    // .then((body) => {
    //     console.log(body)
    //     console.log("HELLEW");
    //     this.setState({predict:body})
    //     // this.setState({ imageURL: `http://localhost:8000/${body.file}` });
    //   })
    // .catch((error) => console.log(error))
    axios.post("http://127.0.0.1:5000/upload", data)
    .then(res => console.log(res))
    .catch(err => console.log(err));
  }

  render() {
    return (
      <form onSubmit={this.handleUploadImage}>
        <div>
          <input ref={(ref) => { this.uploadInput = ref; }} type="file" />
        </div>
        <div>
          <input ref={(ref) => { this.fileName = ref; }}  type="text" placeholder="Enter the desired name of file" />
        </div>
        <br />
        <div>
          <button>Upload</button>
        </div>
        {this.state.predict?
        <div>{this.state.predict}</div>
        :
        null
        }
      </form>
    );
  }
}

export default App;