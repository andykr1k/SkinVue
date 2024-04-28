export default function ScrollingSentences() {
    const sentences1 = [
      "Stay out of the sun as much as possible between 10 a.m. and 4 p.m.",
    ];
    const sentences2 = [
      "Cover up with long sleeves, long pants or a long skirt, a hat, and sunglasses.",
    ];
    const sentences3 = ["Use sunscreen with SPF 15 or higher"];
    
    return (
      <div className="grid h-full text-5xl font-bold p-3 relative  dark:text-ghostwhite text-richblack overflow-hidden">
        <div className="flex animate-scroll1 overflow-hidden w-fit">
          {sentences1.map((sentence, index) => (
            <div key={index} className="whitespace-nowrap">
              {sentence}
            </div>
          ))}
        </div>
        <div className="flex animate-scroll2 overflow-hidden w-fit">
          {sentences2.map((sentence, index) => (
            <div key={index} className="whitespace-nowrap">
              {sentence}
            </div>
          ))}
        </div>
        <div className="flex animate-scroll3 overflow-hidden w-fit">
          {sentences3.map((sentence, index) => (
            <div key={index} className="whitespace-nowrap">
              {sentence}
            </div>
          ))}
        </div>
        <style>
          {`
          .animate-scroll1 {
            animation: scrollText 22s linear infinite;
          }

          .animate-scroll2 {
            animation: scrollText 19s linear infinite;
          }

          .animate-scroll3 {
            animation: scrollText 12s linear infinite;
          }

          @keyframes scrollText {
            0% { transform: translateX(30%); }
            100% { transform: translateX(-100%); }
          }
        `}
        </style>
      </div>
    );
}