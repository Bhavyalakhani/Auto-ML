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
                        <div className="container main-container">
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
                                className="form-control"
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
                                className="form-control"
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
                          {/* <div className="container">
                          <br /><br />
                          <h1>Auto ML</h1>
                          <br /><br />
                          <div className="container">
                            <div className="input-group">
                              <input
                                type="file"
                                className="form-control"
                                id="inputGroupFile04"
                                aria-describedby="inputGroupFileAddon04"
                                aria-label="Upload Dataset"
                              />
                            </div>
                            <label for="exampleInputEmail1" className="form-label"
                              >Target Variable</label
                            >
                            <input
                              type="email"
                              className="form-control"
                              id="exampleInputEmail1"
                              aria-describedby="emailHelp"
                            /><br /><button
                              className="btn btn-outline-secondary"
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
        {/* <footer className="footer has-text-centered">
          <hr></hr>
          <div className="container">
            <p className="has-text-weight-light">
              &copy; 2021 Made by Vatsal, Bhavya, Samit under the guidance of
              Dr. Pratik Kanani
            </p>
          </div>
        </footer> */}
        <footer className="site-footer">
          <div className="container">
            <div className="row">
              <div className="col-sm-12 col-md-12">
                <h6>About</h6>
                <p className="text-justify">
                  Auto ML is an initiative to help the upcoming ML Engineers
                  with training of Models. AutoML focuses on preprocessing and
                  training the most complex datasets without the interference of
                  any human. We will help programmers train models without
                  having any programming knowledge. We will return the model
                  with the best accuracy on the provided dataset.
                </p>
              </div>
            </div>
            <hr></hr>
          </div>
          <div className="container">
            <div className="row">
              <div className="col-md-8 col-sm-6 col-xs-12">
                <p className="copyright-text">
                  Copyright &copy; 2021 All Rights Reserved by
                  <a href="#">The Four Horsemen</a>.
                </p>
              </div>
            </div>
          </div>
        </footer>
      </div>
    );
  }
}

export default App;
