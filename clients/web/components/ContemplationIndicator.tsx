import React, { useState, useEffect } from 'react';

interface ContemplationIndicatorProps {
  emotion: string;
  duration: number; // in milliseconds
  maWeight: number; // 0-1, affects animation style
  maDescription: string;
  indicatorStyle: 'ma_weighted' | 'balanced';
}

export const ContemplationIndicator: React.FC<ContemplationIndicatorProps> = ({
  emotion,
  duration,
  maWeight,
  maDescription,
  indicatorStyle
}) => {
  const [progress, setProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.min((elapsed / duration) * 100, 100);
      setProgress(newProgress);

      if (newProgress >= 100) {
        clearInterval(interval);
        // Fade out after completion
        setTimeout(() => setIsVisible(false), 500);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [duration]);

  // Ma-weighted styling - higher Ma = more contemplative, spacious
  const getMaWeightedStyles = () => {
    if (indicatorStyle === 'ma_weighted' && maWeight > 0.4) {
      return {
        animationDuration: `${3 + maWeight * 2}s`, // Slower breathing for high Ma
        opacity: 0.9 - (maWeight * 0.2), // More subtle for high Ma
        fontSize: `${0.9 + maWeight * 0.2}rem`, // Slightly larger text
      };
    }
    return {
      animationDuration: '2.5s',
      opacity: 1,
      fontSize: '1rem',
    };
  };

  const getContemplationMessage = (emotion: string, maDescription: string) => {
    // High Ma (contemplative) messages
    if (maWeight > 0.5) {
      const highMaMessages = {
        grief: "Oviya is sitting with you in this sacred heaviness...",
        loss: "Oviya is holding this profound space of loss with you...",
        vulnerability: "Oviya is receiving your trust in this gentle space...",
        shame: "Oviya is holding non-judgmental space for your heart...",
        sadness: "Oviya is feeling the depth of this sadness alongside you...",
        anxiety: "Oviya is creating a calm, spacious container for you...",
        anger: "Oviya is listening deeply to your strong feelings...",
        joy: "Oviya is holding space for your joy to expand...",
        default: "Oviya is deeply present with you...",
      };
      return highMaMessages[emotion] || highMaMessages.default;
    }

    // Medium Ma (balanced) messages
    else if (maWeight > 0.3) {
      const mediumMaMessages = {
        grief: "Oviya is sitting with you in this heaviness...",
        loss: "Oviya is holding space for your grief...",
        vulnerability: "Oviya is receiving your trust with care...",
        shame: "Oviya is meeting your vulnerability gently...",
        sadness: "Oviya is feeling the weight of this with you...",
        anxiety: "Oviya is creating a calm space for you...",
        anger: "Oviya is listening to your strong feelings...",
        joy: "Oviya is celebrating this moment with you...",
        default: "Oviya is present with you...",
      };
      return mediumMaMessages[emotion] || mediumMaMessages.default;
    }

    // Low Ma (direct) messages
    else {
      const lowMaMessages = {
        grief: "Oviya is here with you...",
        loss: "Oviya is supporting you...",
        vulnerability: "Oviya hears you...",
        shame: "Oviya is here...",
        sadness: "Oviya understands...",
        anxiety: "Oviya is listening...",
        anger: "Oviya hears your feelings...",
        joy: "Oviya shares your joy...",
        default: "Oviya is thinking...",
      };
      return lowMaMessages[emotion] || lowMaMessages.default;
    }
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      grief: 'from-slate-600 to-slate-800',
      loss: 'from-slate-600 to-slate-800',
      vulnerability: 'from-amber-600 to-amber-800',
      shame: 'from-amber-600 to-amber-800',
      sadness: 'from-blue-600 to-blue-800',
      anxiety: 'from-blue-500 to-blue-700',
      anger: 'from-red-600 to-red-800',
      joy: 'from-yellow-400 to-yellow-600',
      default: 'from-purple-500 to-purple-700'
    };
    return colors[emotion] || colors.default;
  };

  if (!isVisible) return null;

  const maStyles = getMaWeightedStyles();
  const message = getContemplationMessage(emotion, maDescription);
  const colorClasses = getEmotionColor(emotion);

  return (
    <div className={`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50
                     transition-opacity duration-500 ${progress >= 100 ? 'opacity-0' : 'opacity-100'}`}>
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 max-w-md mx-auto
                      border border-gray-200/50">
        {/* Ma-weighted breathing circle */}
        <div className="flex justify-center mb-6">
          <div
            className={`w-16 h-16 rounded-full bg-gradient-to-br ${colorClasses}
                        animate-pulse shadow-lg`}
            style={{
              animationDuration: maStyles.animationDuration,
              opacity: maStyles.opacity
            }}
          />
        </div>

        {/* Contemplation message */}
        <div className="text-center mb-6">
          <p
            className="text-gray-800 font-medium leading-relaxed"
            style={{
              fontSize: maStyles.fontSize,
              opacity: maStyles.opacity
            }}
          >
            {message}
          </p>
          {maWeight > 0.4 && (
            <p className="text-xs text-gray-500 mt-2 italic">
              {maDescription}
            </p>
          )}
        </div>

        {/* Progress indicator */}
        <div className="w-full bg-gray-200 rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${colorClasses} transition-all duration-100 ease-out rounded-full`}
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Optional: Ma weight indicator for development */}
        {process.env.NODE_ENV === 'development' && (
          <div className="mt-4 text-xs text-gray-400 text-center">
            Ma: {(maWeight * 100).toFixed(0)}% • Duration: {Math.round(duration / 1000)}s
          </div>
        )}
      </div>
    </div>
  );
};

export default ContemplationIndicator;
