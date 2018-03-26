using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.AI.MachineLearning.Preview;
using Windows.Foundation.Collections;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.UI;

using WinMLExplorer.Models;

namespace WinMLExplorer.MLModels
{
    public sealed class BirdModelInput
    {
        public VideoFrame data { get; set; }
    }

    public sealed class BirdModelOutput
    {
        public IList<string> classLabel { get; set; }
        public IDictionary<string, float> loss { get; set; }
        public BirdModelOutput()
        {
            this.classLabel = new List<string>();
            this.loss = new Dictionary<string, float>()
            {
                { "cockatoo", 0f },
                { "fantail", 0f },
                { "kaka", 0f },
                { "kakapo", 0f },
                { "kea", 0f },
                { "kiwi", 0f },
                { "kokako", 0f },
                { "morepork", 0f },
                { "tui", 0f },
            };
        }
    }

    public sealed class BirdModel : WinMLModel
    {
        public override string DisplayInputName => "Native Birds";

        public override float DisplayMinProbability => 0.1f;

        public override string DisplayName => "Bird Detection";

        public override DisplayResultSetting[] DisplayResultSettings => new DisplayResultSetting[]
        {
            new DisplayResultSetting() { Name = "cockatoo", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "fantail", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "kaka", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "kakapo", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "kea", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "kiwi", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "kokako", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "morepork", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "tui", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) }
        };

        public override string Filename => "birds.onnx";

        public override string Foldername => "Birds";

        protected override async Task EvaluateAsync(MLModelResult result, VideoFrame inputFrame)
        {
            // Initialize the input
            BirdModelInput input = new BirdModelInput() { data = inputFrame };

            // Evaluate the input
            BirdModelOutput output = await EvaluateAsync(input, result.CorrelationId);

            // Get first label from output
            string label = output.classLabel?.FirstOrDefault();

            // Find probability for label
            if (string.IsNullOrEmpty(label) == false)
            {
                float probability = output.loss?.ContainsKey(label) == true ? output.loss[label] : 0f;

                result.OutputFeatures = new MLModelOutputFeature[]
                {
                    new MLModelOutputFeature() { Label = label, Probability = probability }
                };
            }
        }

        public async Task<BirdModelOutput> EvaluateAsync(BirdModelInput input, string correlationId = "")
        {
            BirdModelOutput output = new BirdModelOutput();

            // Bind input and output model
            LearningModelBindingPreview binding = new LearningModelBindingPreview(this.LearningModel);
            binding.Bind("data", input.data);
            binding.Bind("classLabel", output.classLabel);
            binding.Bind("loss", output.loss);

            // Evaluate the bindings
            LearningModelEvaluationResultPreview evalResult = await this.LearningModel.EvaluateAsync(binding, correlationId);
            return output;
        }
    }
}
