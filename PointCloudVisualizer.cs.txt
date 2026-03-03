using System.Collections.Generic;

using UnityEngine;
using UnityEngine.VFX;
using PointCloudScanner;
using System;


/// <summary>
/// Implements point cloud visualization using VFX Graph.
///
/// The point cloud is implemented as a Unity Visual Effect (PointCloud.vfx).
/// It takes some properties and renders particles (in this case points)
/// based on them. This script is responsible for updating those properties.
/// More details can be found by opening the effect in Unity editor.
///
/// The implementation is based on the following tutorial, with some
/// modifications to fit our needs: https://www.youtube.com/watch?v=P5BgrdXis68
///
/// E.g. Texture2D was replaced with a GraphicsBuffer for more performant writes.
/// </summary>
public class PointCloudVisualizer : MonoBehaviour
{
    public PointCloudManager PointCloudManager;
    public static bool Disable = false;
    private VisualEffect _pointCloud;
    private GraphicsBuffer _buffPoints;
    private bool _updatePending = false;
    private uint _particleCount = 0;
    private int _index = 0; // Current index in the points buffer
    private const int BuffStride = 3 * sizeof(float) + 4; // Point is Vector3 + rgba packed in uint (like Color32)

    // These [SerializeField] + public getters/setters allow doing necessary
    // updates when the values are modified in the Inspector.
    [SerializeField] private uint _maxPoints = 2_000_000;
    public uint MaxPoints
    {
        get => _maxPoints;
        set
        {
            // Hard limits to prevent bad things from happening.
            // Would be even safer if the host was queried for max buffer size (if possible).
            value = Math.Clamp(value, 1, 100_000_000);

            _maxPoints = value;

            // Resize the buffer
            if (PointCloudManager != null && PointCloudManager.Points != null)
            {
                var points = PointCloudManager.Points;
                if (_buffPoints != null)
                {
                    if (_buffPoints.count == value)
                        return;
                    _buffPoints.Dispose();
                }
                _buffPoints = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)value, BuffStride);
                var count = Math.Min(points.Count, (int)value);
                if (count > 0)
                    _buffPoints.SetData(points, points.Count - count, 0, count);
                _index = count;
                _particleCount = (uint)count;
                _updatePending = true;
            }
        }
    }
    [SerializeField] private float _particleSize = 0.02f;
    public float ParticleSize
    {
        get => _particleSize;
        set
        {
            _particleSize = Math.Clamp(value, 0.001f, 1.0f);
            _updatePending = true;
        }
    }
    [SerializeField] private float _brightness = 2.75f;
    public float Brightness
    {
        get => _brightness;
        set
        {
            _brightness = Math.Clamp(value, 1f, 10.0f);
            _updatePending = true;
        }
    }

    /// <summary>
    /// Called when Unity is loaded or a value changes in the Inspector.
    /// </summary>
    private void OnValidate()
    {
        if (Disable)
            return;
        // Call setters when the values are modified in the Inspector.
        MaxPoints = _maxPoints;
        ParticleSize = _particleSize;
    }

    /// <summary>
    /// Called when enabled, just before first Update.
    /// </summary>
    private void Start()
    {
        if (Disable)
            return;
        _pointCloud = GetComponent<VisualEffect>();
        if (_buffPoints == null)
            _buffPoints = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)_maxPoints, BuffStride);
    }

    /// <summary>
    /// Called when enabled.
    /// </summary>
    private void OnEnable()
    {
        if (Disable)
            return;
        if (_buffPoints == null)
            _buffPoints = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)_maxPoints, BuffStride);
        else
            Debug.LogWarning($"{nameof(_buffPoints)} somehow already exists after enabling.");
    }

    /// <summary>
    /// Called when disabled.
    /// </summary>
    private void OnDisable()
    {
        if (Disable)
            return;
        _buffPoints.Dispose();
        _buffPoints = null;
        _particleCount = 0;
    }

    /// <summary>
    /// Called once per frame.
    /// </summary>
    private void Update()
    {
        if (Disable)
            return;
        if (_pointCloud == null)
        {
            Debug.LogWarning("No pointcloud vfx");
            return;
        }

        if (_updatePending)
        {
            _updatePending = false;
            _pointCloud.Reinit();
            // Update effect properties
            _pointCloud.SetUInt(Shader.PropertyToID("ParticleCount"), _particleCount);
            _pointCloud.SetGraphicsBuffer(Shader.PropertyToID("BuffPoints"), _buffPoints);
            _pointCloud.SetFloat(Shader.PropertyToID("ParticleSize"), _particleSize);
            _pointCloud.SetFloat(Shader.PropertyToID("Brightness"), _brightness);
        }
    }

    /// <summary>
    /// Adds given points to the point cloud.
    /// Once max points is reached, oldest points are overwritten circularly.
    /// <param name="points">The points to add.</param>
    /// </summary>
    public void AddPoints(List<Point> points)
    {
        if (Disable)
            return;

        if (points.Count == 0)
            return;

        // Larger or equal amount of data as the size of the graphics buffer.
        // Just copy the data at the end (most recent data) to the buffer.
        //
        // Data    ######################
        // Buffer        [               ]
        if (points.Count >= _maxPoints)
        {
            _buffPoints.SetData(points, points.Count - (int)_maxPoints, 0, (int)_maxPoints);
            _particleCount = _maxPoints;
            _index = 0;
        }
        // Less data than the buffer size, but requires wrap around.
        // Write in 2 parts.
        //
        //                Part 2  Part 1
        // Data           ######  #######
        // Buffer        [               ]
        // Start index            ^
        else if (_index + points.Count >= _maxPoints)
        {
            int part1Count = (int)_maxPoints - _index;
            _buffPoints.SetData(points, 0, _index, part1Count);

            int part2Count = points.Count - part1Count;
            _buffPoints.SetData(points, part1Count, 0, part2Count);

            _index = part2Count;
        }
        // No wrap around required.
        // Single write is enough.
        //
        // Data                   #######
        // Buffer        [               ]
        // Start index            ^
        else
        {
            _buffPoints.SetData(points, 0, _index, points.Count);
            _particleCount = Math.Max(_particleCount, (uint)_index + (uint)points.Count);
            _index = (_index + points.Count) % (int)_maxPoints;
        }

        _updatePending = true;
    }
}
