import React, { useState, useEffect } from "react";

export default function FlipFacts() {
  const [index, setIndex] = useState(0);
  const data = [
    {
      percentage: 32,
      fact: "increased number of new invasive melanoma cases diagnosed within the past decade",
    },
    {
      percentage: 20,
      fact: "of Americans will develop skin cancer by the age of 70",
    },
    {
      percentage: 55,
      fact: "higher change for men to die of melanoma, from ages 15 to 39, compared to their female counterparts",
    },
    {
      percentage: 80,
      fact: "of the sun’s harmful UV rays can penetrate clouds on cloudy days",
    },
    {
      percentage: 86,
      fact: "of melanoma cases can be attributed to exposure to ultraviolet radiation from the sun",
    },
  ];
    useEffect(() => {
      const interval = setInterval(() => {
        setIndex((prevIndex) => (prevIndex + 1) % data.length);
      }, 5000);

      return () => clearInterval(interval);
    }, [data.length]);


  const { percentage, fact } = data[index];

  return (
    <div className="grid place-items-center h-full dark:text-ghostwhite text-richblack p-2">
      <h2 className="text-xl font-bold text-center">
        {percentage}% {fact}
      </h2>
    </div>
  );
}
