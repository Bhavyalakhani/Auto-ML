import React from 'react';
import axios from 'axios';
class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      imageURL: '',
      predict: null,
      modelname: null,
    };

    this.handleUploadImage = this.handleUploadImage.bind(this);
  }

  handleUploadImage(ev) {
    console.log('FUNCTION CLLED');
    ev.preventDefault();
    var myHeaders = new Headers();
    myHeaders.append('Content-Type', 'multipart/form-data');
    const data = new FormData();
    console.log(this.uploadInput.files[0].name);
    data.append('file', this.uploadInput.files[0]);
    data.append('target', this.fileName.value);
    console.log(this.fileName.value);
    // data.append('file', this.uploadInput.files[0]);
    // data.append('filename', this.fileName.value);
    console.log(data, this.uploadInput.files[0]);
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
    axios
      .post('http://127.0.0.1:5000/upload', data, {
        headers: {
          Accept: 'application/json',
        },
      })
      .then((res) => {
        console.log(res);
        this.setState((state) => ({
          predict: Number(res.data.output) * 100,
          modelname: res.data.model,
        }));
      })
      .catch((err) => console.log(err));
  }

  render() {
    return (
      <div>
        <nav className="navbar navbar-expand-lg navbar-light bg-light">
          <div className="container">
            <a className="navbar-brand" href="#">
              Auto ML
            </a>
            <button
              className="navbar-toggler"
              type="button"
              data-toggle="collapse"
              data-target="#navbarNav"
              aria-controls="navbarNav"
              aria-expanded="false"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarNav">
              <ul className="navbar-nav">
                <li className="nav-item active">
                  <a className="nav-link" href="#">
                    Home
                  </a>
                </li>
                <li className="nav-item">
                  <a className="nav-link" href="#">
                    Features
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        <div className="container">
          <div className="container">
            <div className="container">
              <div className="container">
                <div className="container">
                  <div className="container">
                    <div className="container">
                      <div className="container">
                        <div className="container">
                          <br></br>
                          {/* <h1>Auto ML</h1> */}
                          <br></br>
                          <form onSubmit={this.handleUploadImage}>
                            <div>
                              <label>Choose a dataset</label>
                              <br></br>
                              {/* <input
                                ref={(ref) => {
                                  this.uploadInput = ref;
                                }}
                                type="file"
                              /> */}
                              <input
                                type="file"
                                ref={(ref) => {
                                  this.uploadInput = ref;
                                }}
                                class="form-control"
                                id="inputGroupFile04"
                                aria-describedby="inputGroupFileAddon04"
                                aria-label="Upload Dataset"
                              />
                            </div>
                            <br></br>
                            <div>
                              {/* <input
                                ref={(ref) => {
                                  this.fileName = ref;
                                }}
                                type="text"
                                placeholder="Enter the name of Target Variable"
                              /> */}
                              <input
                                type="text"
                                class="form-control"
                                id="exampleInputEmail1"
                                aria-describedby="emailHelp"
                                placeholder="Enter the name of Target Variable"
                                ref={(ref) => {
                                  this.fileName = ref;
                                }}
                              />
                            </div>
                            <br />
                            <div>
                              <button>Upload</button>
                            </div>
                          </form>
                          <div>
                            {this.state.modelname ? (
                              <>
                                <h1>Results</h1>
                                <div>
                                  Model with the best accuracy:{' '}
                                  <b>{this.state.modelname}</b>
                                </div>
                              </>
                            ) : null}
                            {this.state.predict ? (
                              <div>
                                Accuracy: <b>{this.state.predict}%</b>
                              </div>
                            ) : null}
                          </div>
                          {/* <div class="container">
                          <br /><br />
                          <h1>Auto ML</h1>
                          <br /><br />
                          <div class="container">
                            <div class="input-group">
                              <input
                                type="file"
                                class="form-control"
                                id="inputGroupFile04"
                                aria-describedby="inputGroupFileAddon04"
                                aria-label="Upload Dataset"
                              />
                            </div>
                            <label for="exampleInputEmail1" class="form-label"
                              >Target Variable</label
                            >
                            <input
                              type="email"
                              class="form-control"
                              id="exampleInputEmail1"
                              aria-describedby="emailHelp"
                            /><br /><button
                              class="btn btn-outline-secondary"
                              type="button"
                              id="inputGroupFileAddon04"
                            >
                              Submit
                            </button>
                          </div>
                        </div> */}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
