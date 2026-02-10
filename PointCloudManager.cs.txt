using UnityEngine;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using UnityEngine.SceneManagement;
using System.Threading.Tasks;
using UnityEngine.VFX;
using NUnit.Framework;

namespace PointCloudScanner
{
    /// <summary>
    /// Represents a point in the point cloud with position and color.
    /// </summary>
    [VFXType(VFXTypeAttribute.Usage.GraphicsBuffer)]
    public struct Point
    {
        public Vector3 Position;
        public uint RGBA; // Color32 can't be used here due to it not being VFX compatible.

        public Color32 Color
        {
            readonly get
            {
                byte r = (byte)(RGBA & 0xFF);
                byte g = (byte)((RGBA >> 8) & 0xFF);
                byte b = (byte)((RGBA >> 16) & 0xFF);
                byte a = (byte)((RGBA >> 24) & 0xFF);
                return new Color32(r, g, b, a);
            }
            set
            {
                RGBA = (uint)(value.r | (value.g << 8) | (value.b << 16) | (value.a << 24));
            }
        }

        public Point(Vector3 position, Color32 color)
        {
            this.Position = position;
            RGBA = (uint)(color.r | (color.g << 8) | (color.b << 16) | (color.a << 24));
        }
    }

    /// <summary>
    /// Manages the point cloud scanning, saving, and loading operations.
    /// See in-code comments for specifics.
    /// </summary>
    public class PointCloudManager : MonoBehaviour
    {
        public int NumRays; // Increase for higher accuracy
        public float MaxDistance; // Max scan range
        public Material PointMaterial; // Assign in Inspector
        public float CameraRotationThreshold; // Threshold for camera rotation
        public float CameraTranslationThreshold; // Threshold for camera translation
        public GameObject PointCloud;
        public Material highlightMaterial;  // Material for highlighting edges
        internal PointCloudVisualizer _visualizer;
        internal List<Point> _points; // All scanned points
        public List<Point> Points { get => _points; }
        public StreamWriter _pcStream; // StreamWriter for writing pointclouds to files
        public StreamWriter _cameraStream; // StreamWriter for writing camera data to files
        public StringBuilder _pcStringBuilder; // StringBuilder for accumulating pointcloud data
        public StringBuilder _cameraStringBuilder; // StringBuilder for accumulating camera data
        internal string _filenameIdentifier; // File path for saving data
        internal float _flushStringsTime; // Time interval for flushing strings to files
        internal float _flushStringsTimer; // Timer for flushing strings to files
        internal Pose _lastCameraPose; // Last camera pose
        internal Camera _mainCamera; // Main camera
        internal bool _isFlushComplete; // Add the IsFlushComplete property
        public List<Point> _scannedPoints; // Current scanned points
        internal bool _displayPoints = true; // Display points in the scene
        internal bool _saved = false; // Whether data has been saved to disk on exit
        internal Camera _sensorCam;
        public GameObject SensorCamObj;
        public RenderTexture _rt;
        readonly private int _camWidth = 84;
        readonly private int _camHeight = 48;
        public static bool DisableCamColor = false;

        private Keybinds _keybinds;

        public bool IsFlushComplete { get; private set; }

        /// <summary>
        /// Sets up references to other objects.
        /// </summary>
        private void Awake()
        {
            if (PointCloud != null)
            {
                _visualizer = PointCloud.GetComponent<PointCloudVisualizer>();
            }
            else
            {
                Debug.LogWarning("PointCloud GameObject is missing, visualization won't work!");
            }

            _mainCamera = Camera.main;

            if (!DisableCamColor)
            {
                _rt = new RenderTexture(_camWidth, _camHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                _rt.name = "Sensor RenderTexture";
                _rt.useMipMap = false;
                _rt.autoGenerateMips = false;
                _rt.filterMode = FilterMode.Point;
                _rt.hideFlags = HideFlags.DontSave;

                if (!SensorCamObj)
                    SensorCamObj = new GameObject();

                _sensorCam = SensorCamObj.AddComponent<Camera>();
                _sensorCam.name = "Sensor camera";
                _sensorCam.renderingPath = RenderingPath.Forward;
                _sensorCam.clearFlags = CameraClearFlags.SolidColor;
                _sensorCam.backgroundColor = Color.black;
                _sensorCam.fieldOfView = _mainCamera.fieldOfView;
                _sensorCam.targetTexture = _rt;
                _sensorCam.useOcclusionCulling = false;
                _sensorCam.allowHDR = false;
                _sensorCam.allowMSAA = false;

                _sensorCam.enabled = false;
            }

            _keybinds = KeybindManager.Instance.keybinds;
        }

        private SeenMeshManager _seenMeshManager;

        /// <summary>
        /// Runs when the behavior is destroyed
        /// </summary>
        private void OnDestroy()
        {
            if (_saved)
                return;

            FinalFlush();

            _saved = true;
        }

        /// <summary>
        /// Runs when the behavior becomes disabled
        /// </summary>
        private void OnDisable()
        {
            if (_saved)
                return;

            FinalFlush();

            _saved = true;
        }

        /// <summary>
        /// Initializes the point cloud manager and sets up file paths.
        /// </summary>
        private void Start()
        {
            // Initialize SeenMeshManager
            _seenMeshManager = new SeenMeshManager();
            _seenMeshManager.SetMeshMaterial(highlightMaterial);

            // Initiate files for streaming data
            string savetime = StaticVariables.starttime;

            // Check if built on macOS, if so, set the path accordingly
            string folderPath = Application.platform == RuntimePlatform.OSXPlayer ? "../../PointCloudDataSets" : "../PointCloudDataSets";
            
            // Set the save directory to be outside the datapath
            string customSaveDirectory = Path.Combine(Application.dataPath, folderPath);

            // Ensure the base directory exists
            if (!Directory.Exists(customSaveDirectory))
            {
                Directory.CreateDirectory(customSaveDirectory);
                Debug.Log($"Directory created at {customSaveDirectory}");
            }

            // Create a subdirectory for the current session
            string sessionDirectory = Path.Combine(customSaveDirectory, savetime + "_" + StaticVariables.identifier);
            if (!Directory.Exists(sessionDirectory))
            {
                Directory.CreateDirectory(sessionDirectory);
                Debug.Log($"Session directory created at {sessionDirectory}");
            }

            // Set the filename identifier
            _filenameIdentifier = savetime + "_" + StaticVariables.identifier + "_" + SceneManager.GetActiveScene().name + "_";

            // Point cloud file
            string pcpath = Path.Combine(sessionDirectory, _filenameIdentifier + "pointcloudstr.txt");
            File.WriteAllText(pcpath, string.Empty); // Ensure the file is created
            _pcStream = new StreamWriter(pcpath, true);
            _pcStringBuilder = new StringBuilder();

            // Camera file
            string cameraPath = Path.Combine(sessionDirectory, _filenameIdentifier + "camerastr.txt");
            File.WriteAllText(cameraPath, string.Empty); // Ensure the file is created
            _cameraStream = new StreamWriter(cameraPath, true);
            _cameraStringBuilder = new StringBuilder();

            // Initialize other variables
            if (NumRays == 0) NumRays = 50;
            if (MaxDistance == 0.0f) MaxDistance = 10.0f;
            CameraRotationThreshold = 0.9f;
            CameraTranslationThreshold = 0.1f;
            _flushStringsTime = 15.0f;
            _flushStringsTimer = 15.0f;
            _points = new List<Point>();
            _scannedPoints = new List<Point>();
        }

        /// <summary>
        /// Updates the point cloud manager, handles timers, scanning and outputting points to the file.
        /// </summary>
        private void Update()
        {
            // Flush strings timer
            _flushStringsTimer -= Time.deltaTime;

            if (_flushStringsTimer <= 0.0f)
            {
                _flushStringsTimer = _flushStringsTime;
                FlushStringBuilders();
            }
            if (ShouldAccumulatePoints())
            {
                ScanObject();
                _lastCameraPose = new Pose(_mainCamera.transform.position, _mainCamera.transform.rotation);
            }
            // Toggle display points
            if (Input.GetKeyDown(_keybinds.TogglePointCloud))
                TogglePointCloudDisplay();
            // Toggle visibility of the seen mesh
            if (Input.GetKeyDown(_keybinds.ToggleSeenMesh))
                _seenMeshManager.ToggleVisibility();
        }

        /// <summary>
        /// Determines if points should be accumulated based on camera movement.
        /// </summary>
        private bool ShouldAccumulatePoints()
        {
            var cameraTransform = _mainCamera.transform;
            return _points.Count == 0
                || Vector3.Dot(_lastCameraPose.forward, cameraTransform.forward) <= CameraRotationThreshold
                || (_lastCameraPose.position - cameraTransform.position).sqrMagnitude >= CameraTranslationThreshold;
        }

        /// <summary>
        /// Renders and handles GUI events, currently not used but can be uncommented for debugging.
        /// </summary>
        private void OnGUI()
        {
            // Debug: draw the sensor camera output on screen
            // if (_rt != null)
            //     GUI.DrawTexture(new Rect(10, 10, 256, 256), _rt, ScaleMode.ScaleToFit);
        }

        /// <summary>
        /// Scans the environment and collects points for the point cloud.
        /// </summary>
        public void ScanObject()
        {
            _scannedPoints.Clear();

            if (_mainCamera == null)
            {
                Debug.LogError("Main camera not found.");
                return;
            }

            Texture2D camTex = null;
            if (!DisableCamColor)
            {
                _sensorCam.transform.position = _mainCamera.transform.position;
                _sensorCam.transform.rotation = _mainCamera.transform.rotation;
                _sensorCam.cullingMask = LayerMask.GetMask("Vegetation", "Terrain");
                _sensorCam.farClipPlane = 1.5f * MaxDistance;
                _sensorCam.fieldOfView = _mainCamera.fieldOfView;
                _sensorCam.Render();
                RenderTexture.active = _sensorCam.targetTexture;

                // CPU texture
                camTex = new Texture2D(_camWidth, _camHeight, TextureFormat.ARGB32, false);
                camTex.filterMode = FilterMode.Point;
                camTex.ReadPixels(new Rect(0, 0, _camWidth, _camHeight), 0, 0);
            }

            for (int i = 0; i < NumRays; i++)
            {
                Vector2 screenPos = new Vector2(UnityEngine.Random.value, UnityEngine.Random.value);
                Ray ray = Camera.main.ViewportPointToRay(screenPos);
                if (Physics.Raycast(ray, out RaycastHit hit, MaxDistance))
                {
                    if (hit.collider.gameObject.name == "Player" ||
                        hit.collider.gameObject.name == "SeenMesh" ||
                        hit.collider.gameObject.name == "UfoFinal(Clone)")
                    {
                        continue;
                    }

                    // Check if the object has an LODGroup
                    var lodGroup = hit.collider.GetComponentInParent<LODGroup>();
                    MeshFilter meshFilter = null;

                    if (lodGroup != null)
                    {
                        // Get the LOD0 meshes
                        var lods = lodGroup.GetLODs();
                        if (lods.Length > 0 && lods[0].renderers != null)
                        {
                            float closestDistance = float.MaxValue;

                            // Find the closest mesh in LOD0
                            foreach (var lodRenderer in lods[0].renderers)
                            {
                                if (lodRenderer is MeshRenderer meshRenderer)
                                {
                                    var mf = meshRenderer.GetComponent<MeshFilter>();
                                    if (mf != null)
                                    {
                                        float distance = Vector3.Distance(hit.point, mf.transform.position);
                                        if (distance < closestDistance)
                                        {
                                            closestDistance = distance;
                                            meshFilter = mf;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // Fallback to the hit object's MeshFilter
                        meshFilter = hit.collider.GetComponent<MeshFilter>();
                    }

                    if (meshFilter != null)
                    {
                        // Highlight the triangle edges of the current hit triangle
                        _seenMeshManager.AddTriangleToSeenList(meshFilter, hit);
                    }

                    Color color = camTex != null ? camTex.GetPixelBilinear(screenPos.x, screenPos.y) : Color.black;
                    Point newPoint = new Point(hit.point, color);
                    _points.Add(newPoint);
                    _scannedPoints.Add(newPoint);
                }
            }

            SavePointCloud();
            if (_displayPoints)
                _visualizer.AddPoints(_scannedPoints);
        }

        /// <summary>
        /// Saves the point cloud to a stringbuilder, which is then written to a file when flushed.
        /// </summary>
        public void SavePointCloud()
        {
            if (Camera.main == null)
            {
                Debug.LogError("Main camera not found.");
                return;
            }

            float timestamp = Time.time;
            Vector3 cameraPosition = Camera.main.transform.position;
            Vector3 cameraAngle = Camera.main.transform.eulerAngles;

            // Save camera data once per scan
            _cameraStringBuilder.AppendLine(
                (
                    (-cameraPosition.x)
                    + " "
                    + cameraPosition.y
                    + " "
                    + cameraPosition.z
                    + " 0 0 0 "
                    + cameraAngle.x
                    + " "
                    + cameraAngle.y
                    + " "
                    + cameraAngle.z
                    + " "
                    + timestamp
                ).Replace(",", ".")
            );

            // Save each point
            foreach (Point point in _scannedPoints)
            {
                SavePoint(point.Position, point.Color, timestamp);
            }
        }

        /// <summary>
        /// Saves a single point to the point cloud file (when flush happens, not immediately).
        /// </summary>
        public void SavePoint(Vector3 position, Color32 color, float timestamp)
        {
            _pcStringBuilder.AppendLine(
                (
                    (-position.x)
                    + " "
                    + position.y
                    + " "
                    + position.z
                    + " "
                    + color.r
                    + " "
                    + color.g
                    + " "
                    + color.b
                    + " "
                    + timestamp
                ).Replace(",", ".")
            );
        }

        /// <summary>
        /// Flushes the string builders to the files asynchronously.
        /// If not done asynchronously, the main thread becomes blocked, freezing the program.
        /// </summary>
        public void FlushStringBuilders()
        {
            // Check for null fields
            if (_pcStringBuilder == null || _cameraStringBuilder == null)
            {
                Debug.LogError("FlushStringBuilders: StringBuilders are null.");
                return;
            }

            if (_pcStream == null || _cameraStream == null)
            {
                Debug.LogError("FlushStringBuilders: Streams are null.");
                return;
            }

            // Check if there is data to flush
            if (_pcStringBuilder.Length == 0 && _cameraStringBuilder.Length == 0)
            {
                Debug.LogWarning("FlushStringBuilders: No data to flush.");
                return;
            }

            // Start the coroutine and wait for it to complete
            StartCoroutine(FlushStringBuildersCoroutine());
        }


        /// <summary>
        /// Flushes the string builders to the files synchronously and closes the file streams.
        /// </summary>
        public void FinalFlush()
        {
            Debug.Log("Flushing synchronously and closing.");

            try
            {
                if (_pcStringBuilder.Length > 0)
                {
                    _pcStream.WriteLine(_pcStringBuilder.ToString());
                    _pcStringBuilder.Clear();
                }
                _pcStream.Flush();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error flushing _pcStream: {ex.Message}");
            }
            finally
            {
                _pcStream.Close();
            }

            try
            {
                if (_cameraStringBuilder.Length > 0)
                {
                    _cameraStream.WriteLine(_cameraStringBuilder.ToString());
                    _cameraStringBuilder.Clear();
                }
                _cameraStream.Flush();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error flushing _cameraStream: {ex.Message}");
            }
            finally
            {
                _cameraStream.Close();
            }
        }

        /// <summary>
        /// Toggles the display of the point cloud in the scene.
        /// </summary>
        private void TogglePointCloudDisplay()
        {
            _displayPoints = !_displayPoints;
            PointCloud.SetActive(_displayPoints);

            // The points are cleared when the visualizer is disabled,
            // so all of them need to be added back when re-enabling.
            if (_displayPoints)
                _visualizer.AddPoints(_points);
        }

        /// <summary>
        /// Coroutine for flushing the string builders to the files.
        /// </summary>
        public System.Collections.IEnumerator FlushStringBuildersCoroutine()
        {
            _isFlushComplete = false; // Reset the flag at the start
            Debug.Log("Starting FlushStringBuildersCoroutine.");

            try
            {
                // Write data asynchronously
                if (_pcStringBuilder.Length > 0)
                {
                    _pcStream.WriteLine(_pcStringBuilder.ToString());
                    _pcStringBuilder.Clear();
                }

                if (_cameraStringBuilder.Length > 0)
                {
                    _cameraStream.WriteLine(_cameraStringBuilder.ToString());
                    _cameraStringBuilder.Clear();
                }

                // Flush the streams
                _pcStream.Flush();
                _cameraStream.Flush();
            }
            catch (Exception ex)
            {
                Debug.LogError($"FlushStringBuildersCoroutine encountered an error: {ex.Message}");
            }

            _isFlushComplete = true; // Mark as complete
            Debug.Log("FlushStringBuildersCoroutine completed.");
            yield break;
        }
    }
}